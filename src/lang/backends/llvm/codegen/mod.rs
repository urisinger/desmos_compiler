use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use inkwell::builder::Builder;
use inkwell::module::Module;

use inkwell::types::{BasicType, StructType};
use inkwell::values::FunctionValue;
use inkwell::{AddressSpace, OptimizationLevel};

use crate::expressions::Expressions;
use crate::lang::parser::{Expr, Node};
use functions::IMPORTED_FUNCTIONS;

use self::functions::{free, malloc};

use super::jit::{ExplicitJitFn, ImplicitJitFn};
use super::types::{ListType, ValueType};
use super::value::Value;
use super::{CompiledExpr, CompiledExprs};

mod bin_op;
mod expr;
mod functions;
mod list;

pub fn compile_all_exprs<'ctx>(
    context: &'ctx inkwell::context::Context,
    exprs: &Expressions,
) -> CompiledExprs<'ctx> {
    let mut codegen = CodeGen::new(context, exprs);
    let mut compiled_functions = Vec::new();

    let execution_engine = codegen
        .module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();

    let malloc_type = context
        .ptr_type(AddressSpace::default())
        .fn_type(&[context.i64_type().into()], false);

    let malloc_function = codegen.module.add_function("malloc", malloc_type, None);
    execution_engine.add_global_mapping(&malloc_function, malloc as usize);

    let free_type = context.void_type().fn_type(
        &[
            context.ptr_type(AddressSpace::default()).into(),
            context.i64_type().into(),
        ],
        false,
    );

    let free_function = codegen.module.add_function("free", free_type, None);
    execution_engine.add_global_mapping(&free_function, free as usize);

    IMPORTED_FUNCTIONS
        .into_iter()
        .for_each(|(name, ret, args, p)| {
            let fn_type = ret.type_enum(context).fn_type(
                &args
                    .iter()
                    .map(|arg| arg.metadata(context))
                    .collect::<Vec<_>>(),
                false,
            );

            let function = codegen.module.add_function(name, fn_type, None);
            execution_engine.add_global_mapping(&function, p as usize);
        });

    for (id, expr) in &exprs.exprs {
        match expr {
            Expr::Implicit { lhs, rhs, .. } => {
                let args = [ValueType::Number, ValueType::Number];

                let lhs_name = format!("implicit_{}_lhs", id.0);
                codegen
                    .compile_fn(&lhs_name, lhs, &args)
                    .expect("Failed to compile lhs");
                compiled_functions.push((lhs_name, *id, "implicit_lhs"));

                let rhs_name = format!("implicit_{}_rhs", id.0);
                codegen
                    .compile_fn(&rhs_name, rhs, &args)
                    .expect("Failed to compile rhs");
                compiled_functions.push((rhs_name, *id, "implicit_rhs"));
            }
            Expr::Explicit { expr } => {
                let name = format!("explicit_{}", id.0);

                let args = vec![ValueType::Number];
                codegen
                    .compile_fn(&name, expr, &args)
                    .expect("Failed to compile explicit function");
                compiled_functions.push((name, *id, "explicit"));
            }
            _ => {}
        }
    }

    let mut compiled_exprs = Vec::new();

    for (id, expr) in &exprs.exprs {
        match expr {
            Expr::Implicit { op, .. } => {
                let lhs_name = format!("implicit_{}_lhs", id.0);
                let lhs = unsafe {
                    ImplicitJitFn::from_function(
                        &lhs_name,
                        &execution_engine,
                        codegen.return_types[&lhs_name],
                    )
                }
                .expect("Failed to retrieve compiled lhs");

                let rhs_name = format!("implicit_{}_rhs", id.0);
                let rhs = unsafe {
                    ImplicitJitFn::from_function(
                        &rhs_name,
                        &execution_engine,
                        codegen.return_types[&rhs_name],
                    )
                }
                .expect("Failed to retrieve compiled rhs");

                compiled_exprs.push(CompiledExpr::Implicit { lhs, op: *op, rhs });
            }
            Expr::Explicit { .. } => {
                let name = format!("explicit_{}", id.0);

                let lhs = unsafe {
                    ExplicitJitFn::from_function(
                        &name,
                        &execution_engine,
                        codegen.return_types[&name],
                    )
                }
                .expect("Failed to retrieve compiled explicit function");

                compiled_exprs.push(CompiledExpr::Explicit { lhs });
            }
            _ => {}
        }
    }

    codegen.module.print_to_stderr();

    CompiledExprs {
        compiled: compiled_exprs,
    }
}

pub struct CodeGen<'ctx, 'expr> {
    pub context: &'ctx inkwell::context::Context,
    pub module: Module<'ctx>,
    pub builder: Builder<'ctx>,
    pub exprs: &'expr Expressions,

    pub return_types: HashMap<String, ValueType>,

    list_type: StructType<'ctx>,
}

impl<'ctx, 'expr> CodeGen<'ctx, 'expr> {
    pub fn new(context: &'ctx inkwell::context::Context, exprs: &'expr Expressions) -> Self {
        let module = context.create_module("main");
        let builder = context.create_builder();

        Self {
            context,
            module,
            builder,
            return_types: HashMap::new(),
            exprs,

            list_type: context.struct_type(
                &[
                    context.i64_type().as_basic_type_enum(),
                    context
                        .ptr_type(AddressSpace::default())
                        .as_basic_type_enum(),
                ],
                true,
            ),
        }
    }

    pub fn get_fn(
        &mut self,
        name: &str,
        types: &[ValueType],
    ) -> Result<(FunctionValue<'ctx>, ValueType)> {
        let len = types.iter().map(|t| t.name().len() + 1).sum::<usize>() + name.len();

        let mut specialized_name = String::with_capacity(len);
        specialized_name.push_str(name);

        for t in types {
            specialized_name.push('_');
            specialized_name.push_str(t.name());
        }

        match self.module.get_function(&specialized_name) {
            Some(function) => Ok((function, self.return_types[name])),
            None => match self.exprs.get_expr(name) {
                Some(Expr::FnDef { rhs, .. }) => {
                    let block = self.builder.get_insert_block();
                    let function = self.compile_fn(&specialized_name, &rhs, types);

                    block.map(|b| self.builder.position_at_end(b));

                    Ok((function?, self.return_types[name]))
                }
                None => bail!("no exprssion found for function {name}"),
                _ => unreachable!("this indicates a bug"),
            },
        }
    }

    pub fn get_var(&mut self, name: &str) -> Result<Value<'ctx>> {
        match self.module.get_function(name) {
            Some(global) => Value::from_basic_value_enum(
                self.builder
                    .build_call(global, &[], name)?
                    .try_as_basic_value()
                    .expect_left("return type should not be void"),
                self.return_types[name],
            )
            .context("type error"),
            None => match self.exprs.get_expr(name) {
                Some(Expr::VarDef { rhs, .. }) => {
                    let block = self.builder.get_insert_block();
                    let compute_global = self.compile_fn(&name, &rhs, &[])?;

                    block.map(|b| self.builder.position_at_end(b));

                    Value::from_basic_value_enum(
                        self.builder
                            .build_call(compute_global, &[], name)?
                            .try_as_basic_value()
                            .expect_left("return type should not be void"),
                        self.return_types[name],
                    )
                    .context("type error")
                }
                None => bail!("no exprssion found for function {name}"),
                _ => unreachable!("this indicates a bug"),
            },
        }
    }

    pub fn compile_fn(
        &mut self,
        name: &str,
        node: &Node,
        args: &[ValueType],
    ) -> Result<FunctionValue<'ctx>> {
        let types: Vec<_> = args.iter().map(|t| t.metadata(&self.context)).collect();

        let ret_type = self.return_type(node, args)?;

        self.return_types.insert(name.to_owned(), ret_type);

        let fn_type = match ret_type {
            ValueType::List(ListType::Number) => self
                .context
                .ptr_type(AddressSpace::default())
                .fn_type(&types, false),
            ValueType::Number => self.context.f64_type().fn_type(&types, false),
        };
        let function = self.module.add_function(name, fn_type, None);
        let args = (0..args.len())
            .map(|i| {
                Value::from_basic_value_enum(
                    function.get_nth_param(i as u32).expect("should not happen"),
                    args[i],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        let value = self.codegen_expr(node, &args)?;

        self.builder
            .build_return(Some(&value.as_basic_value_enum()))?;

        println!(
            "module functions: {:?}",
            self.module
                .get_functions()
                .map(|a| a.get_name().to_str().unwrap().to_string())
                .collect::<Vec<_>>()
        );
        Ok(function)
    }
}