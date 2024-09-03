use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::intrinsics::Intrinsic;
use inkwell::module::Module;

use inkwell::types::StructType;
use inkwell::values::{AnyValue, BasicMetadataValueEnum, BasicValue, FunctionValue};
use inkwell::{AddressSpace, OptimizationLevel};

use crate::expressions::{ExpressionId, Expressions};
use crate::lang::parser::{BinaryOp, ComparisonOp, Expr, Literal, Node, UnaryOp};
use functions::IMPORTED_FUNCTIONS;
use value::{ArrayType, Number, NumberType, Value, ValueType};

pub mod functions;
pub mod value;

#[derive(Debug)]
pub struct CompiledExprs<'ctx> {
    pub compiled: Vec<CompiledExpr<'ctx>>,
}

#[derive(Debug)]
pub enum CompiledExpr<'ctx> {
    Implicit {
        lhs: FunctionValue<'ctx>,
        op: ComparisonOp,
        rhs: FunctionValue<'ctx>,
    },
    Explicit {
        lhs: FunctionValue<'ctx>,
    },
}

pub struct CodeGen<'ctx, 'expr> {
    pub context: &'ctx inkwell::context::Context,
    pub module: Module<'ctx>,
    pub builder: Builder<'ctx>,
    pub execution_engine: ExecutionEngine<'ctx>,
    pub exprs: &'expr Expressions,

    pub list_type: StructType<'ctx>,

    pub imported_functions: HashMap<&'static str, FunctionValue<'ctx>>,
    pub return_types: HashMap<String, ValueType>,
}

impl<'ctx, 'expr> CodeGen<'ctx, 'expr> {
    pub fn new(context: &'ctx inkwell::context::Context, exprs: &'expr Expressions) -> Self {
        let module = context.create_module("main");
        let builder = context.create_builder();

        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let imported_functions = IMPORTED_FUNCTIONS
            .into_iter()
            .map(|(name, ret, args, p)| {
                let fn_type = match ret {
                    NumberType::Float => context.f64_type().fn_type(
                        &args
                            .iter()
                            .map(|arg| arg.metadata(context))
                            .collect::<Vec<_>>(),
                        false,
                    ),

                    NumberType::Int => context.i64_type().fn_type(
                        &args
                            .iter()
                            .map(|arg| arg.metadata(context))
                            .collect::<Vec<_>>(),
                        false,
                    ),
                };

                let function = module.add_function(name, fn_type, None);
                execution_engine.add_global_mapping(&function, p as usize);

                (name, function)
            })
            .collect();

        Self {
            context,
            module,
            builder,
            exprs,
            execution_engine,
            return_types: HashMap::new(),

            imported_functions,
        }
    }

    pub fn get_explicit_fn(
        &self,
        name: &str,
    ) -> Option<JitFunction<'ctx, unsafe extern "C" fn(f64) -> f64>> {
        unsafe { self.execution_engine.get_function(name).ok() }
    }

    pub fn get_implicit_fn(
        &self,
        name: &str,
    ) -> Option<JitFunction<'ctx, unsafe extern "C" fn(f64, f64) -> f64>> {
        unsafe { self.execution_engine.get_function(name).ok() }
    }

    pub fn return_type(&self, expr: &Node, call_types: &[ValueType]) -> Result<ValueType> {
        Ok(match expr {
            Node::Lit(Literal::Int(value)) => ValueType::Number(NumberType::Int),
            Node::Lit(Literal::Float(value)) => ValueType::Number(NumberType::Float),
            Node::Ident(ident) => {
                // Handling identifiers is more complex and often involves looking up values in a symbol table.
                // Here we assume you have some mechanism to resolve identifiers to LLVM values.
                //
                match self
                    .exprs
                    .get_expr(ident)
                    .context(format!("failed to get expr for ident, {}", ident))?
                {
                    Expr::VarDef { ident, rhs } => self.return_type(&rhs, call_types)?,
                    expr => bail!("expr has wrong type, this should not happend, {expr:?}"),
                }
            }
            Node::BinOp { lhs, op, rhs } => {
                let lhs = self.return_type(lhs, call_types)?;
                let rhs = self.return_type(rhs, call_types)?;
                match op {
                    BinaryOp::Pow => ValueType::Number(NumberType::Float),
                    _ => match (lhs, rhs) {
                        (
                            ValueType::Number(NumberType::Int),
                            ValueType::Number(NumberType::Int),
                        ) => ValueType::Number(NumberType::Int),
                        _ => ValueType::Number(NumberType::Float),
                    },
                }
            }
            Node::UnaryOp { val, op } => {
                let val_value = self.return_type(val, call_types)?;

                match op {
                    UnaryOp::Neg => match val_value {
                        ValueType::Number(NumberType::Float) => {
                            ValueType::Number(NumberType::Float)
                        }
                        ValueType::Number(NumberType::Int) => ValueType::Number(NumberType::Int),
                        ValueType::Array(ArrayType::Number(ty)) => match ty {
                            NumberType::Float => {
                                ValueType::Array(ArrayType::Number(NumberType::Float))
                            }
                            NumberType::Int => ValueType::Array(ArrayType::Number(NumberType::Int)),
                        },
                    },
                    _ => unimplemented!(),
                }
            }
            Node::FnCall { ident, args } => match self.exprs.get_expr(ident) {
                Some(Expr::FnDef { rhs, .. }) => {
                    let call_types = args
                        .iter()
                        .map(|arg| {
                            let value = self.return_type(arg, call_types)?;
                            Ok(value)
                        })
                        .collect::<Result<Vec<_>>>()?;

                    self.return_type(&rhs, &call_types)?
                }
                Some(Expr::VarDef { rhs, .. }) => self.return_type(&rhs, call_types)?,
                None => bail!("unknown ident {ident}"),
                _ => bail!("expr has the wrong type"),
            },
            Node::FnArg { index } => call_types[*index],
            _ => unimplemented!(),
        })
    }

    pub fn codegen_expr(
        &self,
        expr: &Node,
        call_args: &[Value<'ctx>],
        block: &BasicBlock,
    ) -> Result<Value<'ctx>> {
        Ok(match expr {
            Node::Lit(Literal::Int(value)) => {
                let int_type = self.context.i64_type();
                int_type.const_int(*value, false).into()
            }
            Node::Lit(Literal::Float(value)) => {
                let float_type = self.context.f64_type();
                float_type.const_float(*value).into()
            }
            Node::Ident(ident) => {
                // Handling identifiers is more complex and often involves looking up values in a symbol table.
                // Here we assume you have some mechanism to resolve identifiers to LLVM values.
                self.get_var(ident)?
            }
            Node::BinOp { lhs, op, rhs } => {
                let lhs = self.codegen_expr(lhs, call_args, block)?;
                let rhs = self.codegen_expr(rhs, call_args, block)?;
                self.codegen_binary_op(lhs, *op, rhs)?
            }
            Node::UnaryOp { val, op } => {
                let val_value = self.codegen_expr(val, call_args, block)?;

                match op {
                    UnaryOp::Neg => match val_value {
                        Value::Number(Number::Float(v)) => {
                            self.builder.build_float_neg(v, "neg")?.into()
                        }
                        Value::Number(Number::Int(v)) => {
                            self.builder.build_int_neg(v, "neg")?.into()
                        }
                    },
                    _ => unimplemented!(),
                }
            }
            Node::FnCall { ident, args } => match self.exprs.get_expr(ident) {
                Some(Expr::FnDef { .. }) => {
                    let (types, args) = args
                        .iter()
                        .map(|arg| {
                            let value = self.codegen_expr(arg, call_args, block)?;
                            Ok((
                                value.get_type(),
                                BasicMetadataValueEnum::from(value.as_basic_value_enum()),
                            ))
                        })
                        .collect::<Result<(Vec<_>, Vec<_>)>>()?;
                    let function = self.get_fn(ident, &types)?;

                    self.builder
                        .build_call(function, &args, ident)?
                        .as_any_value_enum()?
                }
                Some(Expr::VarDef { .. }) => {
                    if args.len() == 1 {
                        self.codegen_binary_op(
                            self.get_var(ident)?,
                            BinaryOp::Mul,
                            self.codegen_expr(&args[0], call_args, block)?,
                        )?
                    } else {
                        bail!("{ident} is not a function")
                    }
                }
                None => bail!("unknown ident {ident}"),
                _ => unreachable!("idents should be VarDef or FnDef only"),
            },
            Node::FnArg { index } => call_args[*index],
            _ => unimplemented!(),
        })
    }

    pub fn codegen_binary_op(
        &self,
        lhs: Value<'ctx>,
        op: BinaryOp,
        rhs: Value<'ctx>,
    ) -> Result<Value<'ctx>> {
        match (lhs, rhs) {
            (Value::Number(lhs), Value::Number(rhs)) => {
                Ok(Value::Number(self.codegen_binary_number_op(lhs, op, rhs)?))
            }
        }
    }

    pub fn codegen_binary_number_op(
        &self,
        lhs: Number<'ctx>,
        op: BinaryOp,
        rhs: Number<'ctx>,
    ) -> Result<Number<'ctx>> {
        Ok(match op {
            BinaryOp::Add => match (lhs, rhs) {
                (Number::Float(lhs), Number::Float(rhs)) => {
                    self.builder.build_float_add(lhs, rhs, "add")?.into()
                }
                (Number::Int(lhs), Number::Int(rhs)) => {
                    self.builder.build_int_add(lhs, rhs, "add")?.into()
                }
                (Number::Int(lhs), Number::Float(rhs)) => self
                    .builder
                    .build_float_add(
                        self.builder.build_signed_int_to_float(
                            lhs,
                            self.context.f64_type(),
                            "cast_int",
                        )?,
                        rhs,
                        "add",
                    )?
                    .into(),

                (Number::Float(lhs), Number::Int(rhs)) => self
                    .builder
                    .build_float_add(
                        lhs,
                        self.builder.build_signed_int_to_float(
                            rhs,
                            self.context.f64_type(),
                            "cast_int",
                        )?,
                        "add",
                    )?
                    .into(),
            },
            BinaryOp::Sub => match (lhs, rhs) {
                (Number::Float(lhs), Number::Float(rhs)) => {
                    self.builder.build_float_sub(lhs, rhs, "add")?.into()
                }
                (Number::Int(lhs), Number::Int(rhs)) => {
                    self.builder.build_int_sub(lhs, rhs, "add")?.into()
                }
                (Number::Int(lhs), Number::Float(rhs)) => self
                    .builder
                    .build_float_sub(
                        self.builder.build_signed_int_to_float(
                            lhs,
                            self.context.f64_type(),
                            "cast_int",
                        )?,
                        rhs,
                        "add",
                    )?
                    .into(),

                (Number::Float(lhs), Number::Int(rhs)) => self
                    .builder
                    .build_float_sub(
                        lhs,
                        self.builder.build_signed_int_to_float(
                            rhs,
                            self.context.f64_type(),
                            "cast_int",
                        )?,
                        "add",
                    )?
                    .into(),
            },
            BinaryOp::Mul => match (lhs, rhs) {
                (Number::Float(lhs), Number::Float(rhs)) => {
                    self.builder.build_float_mul(lhs, rhs, "add")?.into()
                }
                (Number::Int(lhs), Number::Int(rhs)) => {
                    self.builder.build_int_mul(lhs, rhs, "add")?.into()
                }
                (Number::Int(lhs), Number::Float(rhs)) => self
                    .builder
                    .build_float_mul(
                        self.builder.build_signed_int_to_float(
                            lhs,
                            self.context.f64_type(),
                            "cast_int",
                        )?,
                        rhs,
                        "add",
                    )?
                    .into(),

                (Number::Float(lhs), Number::Int(rhs)) => self
                    .builder
                    .build_float_mul(
                        lhs,
                        self.builder.build_signed_int_to_float(
                            rhs,
                            self.context.f64_type(),
                            "cast_int",
                        )?,
                        "add",
                    )?
                    .into(),
            },

            BinaryOp::Div => match (lhs, rhs) {
                (Number::Float(lhs), Number::Float(rhs)) => {
                    self.builder.build_float_div(lhs, rhs, "add")?.into()
                }
                (Number::Int(lhs), Number::Int(rhs)) => {
                    self.builder.build_int_signed_div(lhs, rhs, "add")?.into()
                }
                (Number::Int(lhs), Number::Float(rhs)) => self
                    .builder
                    .build_float_div(
                        self.builder.build_signed_int_to_float(
                            lhs,
                            self.context.f64_type(),
                            "cast_int",
                        )?,
                        rhs,
                        "add",
                    )?
                    .into(),

                (Number::Float(lhs), Number::Int(rhs)) => self
                    .builder
                    .build_float_div(
                        lhs,
                        self.builder.build_signed_int_to_float(
                            rhs,
                            self.context.f64_type(),
                            "cast_int",
                        )?,
                        "add",
                    )?
                    .into(),
            },
            BinaryOp::Pow => match (lhs, rhs) {
                (Number::Float(lhs), Number::Float(rhs)) => {
                    let intrinsic = Intrinsic::find("llvm.pow").unwrap();

                    Number::from_any_value_enum(
                        self.builder
                            .build_call(
                                intrinsic
                                    .get_declaration(
                                        &self.module,
                                        &[
                                            self.context.f64_type().into(),
                                            self.context.f64_type().into(),
                                        ],
                                    )
                                    .unwrap(),
                                &[lhs.into(), rhs.into()],
                                "pow",
                            )?
                            .as_any_value_enum(),
                    )
                    .expect("should be number type")
                }
                (Number::Int(lhs), Number::Int(rhs)) => {
                    let intrinsic = Intrinsic::find("llvm.powi").unwrap();

                    Number::from_any_value_enum(
                        self.builder
                            .build_call(
                                intrinsic
                                    .get_declaration(
                                        &self.module,
                                        &[
                                            self.context.f64_type().into(),
                                            self.context.i64_type().into(),
                                        ],
                                    )
                                    .unwrap(),
                                &[
                                    self.builder
                                        .build_signed_int_to_float(
                                            lhs,
                                            self.context.f64_type(),
                                            "int_to_float",
                                        )?
                                        .into(),
                                    rhs.into(),
                                ],
                                "pow",
                            )?
                            .as_any_value_enum(),
                    )
                    .expect("should be number type")
                }
                (Number::Int(lhs), Number::Float(rhs)) => {
                    let intrinsic = Intrinsic::find("llvm.pow").unwrap();

                    Number::from_any_value_enum(
                        self.builder
                            .build_call(
                                intrinsic
                                    .get_declaration(
                                        &self.module,
                                        &[
                                            self.context.f64_type().into(),
                                            self.context.f64_type().into(),
                                        ],
                                    )
                                    .unwrap(),
                                &[
                                    self.builder
                                        .build_signed_int_to_float(
                                            lhs,
                                            self.context.f64_type(),
                                            "int_to_float",
                                        )?
                                        .into(),
                                    rhs.into(),
                                ],
                                "pow",
                            )?
                            .as_any_value_enum(),
                    )
                    .expect("should be number type")
                }
                (Number::Float(lhs), Number::Int(rhs)) => {
                    let intrinsic = Intrinsic::find("llvm.powi").unwrap();

                    Number::from_any_value_enum(
                        self.builder
                            .build_call(
                                intrinsic
                                    .get_declaration(
                                        &self.module,
                                        &[
                                            self.context.f64_type().into(),
                                            self.context.i64_type().into(),
                                        ],
                                    )
                                    .unwrap(),
                                &[lhs.into(), rhs.into()],
                                "pow",
                            )?
                            .as_any_value_enum(),
                    )
                    .expect("should be number type")
                }
            },
        })
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

    pub fn get_var(&self, name: &str) -> Result<Value<'ctx>> {
        match self.module.get_function(name) {
            Some(global) => self
                .builder
                .build_call(global, &[], name)?
                .as_any_value_enum()
                .try_into(),
            None => match self.exprs.get_expr(name) {
                Some(Expr::VarDef { rhs, .. }) => {
                    let block = self.builder.get_insert_block();
                    let compute_global = self.compile_fn(&name, &rhs, &[])?;

                    block.map(|b| self.builder.position_at_end(b));
                    self.builder
                        .build_call(compute_global, &[], name)?
                        .as_any_value_enum()
                        .try_into()
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

        let fn_type = match ret_type {
            ValueType::Array(ArrayType::Number) => self
                .context
                .ptr_type(AddressSpace::default())
                .fn_type(&types, false),
            ValueType::Number(NumberType::Float) => self.context.f64_type().fn_type(&types, false),
            ValueType::Number(NumberType::Int) => self.context.i64_type().fn_type(&types, false),
        };
        let function = self.module.add_function(name, fn_type, None);
        let args = (0..args.len())
            .map(|i| {
                Value::from_any_value_enum(
                    function
                        .get_nth_param(i as u32)
                        .expect("should not happen")
                        .as_any_value_enum(),
                    args[i],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        let value = self.codegen_expr(node, &args, &entry)?;

        self.return_types.insert(name.to_owned(), value.get_type());
        self.builder
            .build_return(Some(&value.as_basic_value_enum()))?;

        Ok(function)
    }

    pub fn compile_expr(
        &mut self,
        id: ExpressionId,
        expr: &Expr,
        compiled_exprs: &mut Vec<CompiledExpr<'ctx>>,
    ) -> Result<()> {
        match &expr {
            Expr::Explicit { lhs, op, rhs } => {
                let args = [
                    ValueType::Number(NumberType::Float),
                    ValueType::Number(NumberType::Float),
                ];

                let lhs_name = format!("implicit_{}_lhs", id.0);
                let lhs = self.compile_fn(&lhs_name, lhs, &args)?;

                let rhs_name = format!("implicit_{}_rhs", id.0);
                let rhs = self.compile_fn(&rhs_name, rhs, &args)?;

                compiled_exprs.push(CompiledExpr::Implicit { lhs, op: *op, rhs })
            }
            Expr::Implicit { expr } => {
                let name = format!("explicit_{}", id.0);

                let args = vec![ValueType::Number(NumberType::Float)];
                let lhs = self.compile_fn(&name, expr, &args)?;

                compiled_exprs.push(CompiledExpr::Explicit { lhs })
            }
            _ => {}
        };
        Ok(())
    }

    pub fn compile_all_exprs(&self) -> CompiledExprs<'ctx> {
        let mut compiled = Vec::new();

        for (id, expr) in &self.exprs.exprs {
            match self.compile_expr(*id, expr, &mut compiled) {
                Ok(_) => {}
                Err(e) => {
                    println!("err {e} at {id:?}");
                }
            }
        }

        self.module.print_to_stderr();

        CompiledExprs { compiled }
    }
}
