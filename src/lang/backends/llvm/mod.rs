use std::collections::HashMap;

use anyhow::{anyhow, bail, Context, Result};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::intrinsics::Intrinsic;
use inkwell::module::Module;

use inkwell::types::{AnyType, BasicType, StructType};
use inkwell::values::{
    AnyValue, BasicMetadataValueEnum, BasicValue, FunctionValue, IntValue, PointerValue,
};
use inkwell::{AddressSpace, OptimizationLevel};

use crate::expressions::{ExpressionId, Expressions};
use crate::lang::parser::{BinaryOp, ComparisonOp, Expr, Literal, Node, UnaryOp};
use functions::IMPORTED_FUNCTIONS;
use types::{ListType, NumberType, ValueType};
use value::{Number, Value};

use self::functions::malloc;
use self::value::List;

pub mod functions;
pub mod types;
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

    pub imported_functions: HashMap<&'static str, FunctionValue<'ctx>>,
    pub return_types: HashMap<String, ValueType>,

    list_type: StructType<'ctx>,
}

impl<'ctx, 'expr> CodeGen<'ctx, 'expr> {
    pub fn new(context: &'ctx inkwell::context::Context, exprs: &'expr Expressions) -> Self {
        let module = context.create_module("main");
        let builder = context.create_builder();

        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();

        let malloc_type = context
            .ptr_type(AddressSpace::default())
            .fn_type(&[context.i64_type().into()], false);

        let function = module.add_function("malloc", malloc_type, None);
        execution_engine.add_global_mapping(&function, malloc as usize);

        let malloc_type = context.void_type().fn_type(
            &[
                context.ptr_type(AddressSpace::default()).into(),
                context.i64_type().into(),
            ],
            false,
        );

        let function = module.add_function("malloc", malloc_type, None);
        execution_engine.add_global_mapping(&function, malloc as usize);

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

            list_type: context.struct_type(
                &[
                    context.i64_type().as_basic_type_enum(),
                    context
                        .ptr_type(AddressSpace::default())
                        .as_basic_type_enum(),
                ],
                false,
            ),

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
                    UnaryOp::Neg => val_value,
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

    pub fn codegen_expr(&mut self, expr: &Node, call_args: &[Value<'ctx>]) -> Result<Value<'ctx>> {
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
                let lhs = self.codegen_expr(lhs, call_args)?;
                let rhs = self.codegen_expr(rhs, call_args)?;
                self.codegen_binary_op(lhs, *op, rhs)?
            }
            Node::UnaryOp { val, op } => {
                let val_value = self.codegen_expr(val, call_args)?;

                match op {
                    UnaryOp::Neg => match val_value {
                        Value::Number(Number::Float(v)) => {
                            self.builder.build_float_neg(v, "neg")?.into()
                        }
                        Value::Number(Number::Int(v)) => {
                            self.builder.build_int_neg(v, "neg")?.into()
                        }
                        _ => unimplemented!(),
                    },
                    _ => unimplemented!(),
                }
            }
            Node::FnCall { ident, args } => match self.exprs.get_expr(ident) {
                Some(Expr::FnDef { .. }) => {
                    let (types, args) = args
                        .iter()
                        .map(|arg| {
                            let value = self.codegen_expr(arg, call_args)?;
                            Ok((
                                value.get_type(),
                                BasicMetadataValueEnum::from(value.as_basic_value_enum()),
                            ))
                        })
                        .collect::<Result<(Vec<_>, Vec<_>)>>()?;
                    let (function, ret_type) = self.get_fn(ident, &types)?;

                    Value::from_basic_value_enum(
                        self.builder
                            .build_call(function, &args, ident)?
                            .try_as_basic_value()
                            .expect_left("return type should not be void"),
                        ret_type,
                    )
                    .expect("ret type does not match expected ret type")
                }
                Some(Expr::VarDef { .. }) => {
                    if args.len() == 1 {
                        let lhs = self.get_var(ident)?;
                        let rhs = self.codegen_expr(&args[0], call_args)?;
                        self.codegen_binary_op(lhs, BinaryOp::Mul, rhs)?
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

    // Allocate memory for a list of `size` elements
    pub fn codegen_allocate(&self, size: IntValue<'ctx>) -> Result<PointerValue<'ctx>> {
        // Assuming you're using LLVM's `malloc` to allocate memory
        let i64_type = self.context.i64_type();
        let element_size = i64_type.const_int(8, false); // Assuming 8 bytes per element (for f64)
        let total_size = self
            .builder
            .build_int_mul(size, element_size, "total_size")?;

        // Call malloc to allocate memory
        let malloc_fn = self
            .module
            .get_function("malloc")
            .expect("malloc should be defined"); // Assuming `malloc` is defined
                                                 //
                                                 //
        let raw_ptr = self
            .builder
            .build_call(malloc_fn, &[total_size.into()], "malloc_call")?
            .try_as_basic_value()
            .left()
            .expect("return type should not be void")
            .into_pointer_value();

        Ok(raw_ptr)
    }

    pub fn codegen_free(&self, list: List<'ctx>) -> Result<()> {
        match list {
            List::Number(struct_value) => {
                // Assuming the pointer is stored as the first field of the struct
                let pointer_field_index = 1; // Change this if the pointer is at a different index
                let pointer: PointerValue<'ctx> = struct_value
                    .get_field_at_index(pointer_field_index)
                    .expect("Failed to get pointer field")
                    .into_pointer_value();

                // Get the size of the array (assuming the size is stored as the first field)
                let size_field_index = 0; // Change this if the size is at a different index
                let size_value = struct_value
                    .get_field_at_index(size_field_index)
                    .expect("Failed to get size field");

                // Create an integer type for the size
                let int_type = self.context.i64_type(); // Use i32 or i64 as needed

                // Convert size_value to IntValue
                let size_int = size_value.into_int_value();

                // Calculate the total size dynamically: total_size = size * size_of::<f32>()
                let float_size = int_type.const_int(size_of::<f64>() as u64, false); // Use f64 if necessary
                let total_size = self
                    .builder
                    .build_int_mul(size_int, float_size, "total_size")?;

                // Call the free function
                let free_fn = self
                    .module
                    .get_function("free")
                    .expect("Free function not found");
                self.builder
                    .build_call(free_fn, &[pointer.into(), total_size.into()], "free_call");

                Ok(())
            }
        }
    }

    pub fn codegen_list_new(&self, size: IntValue<'ctx>) -> Result<Value<'ctx>> {
        // Allocate memory for the list
        let pointer = self.codegen_allocate(size)?;

        // Create a struct representing the list
        // Assuming the struct contains the size (i32) and the pointer (f64*)
        let list_type = self.context.struct_type(
            &[self.context.i32_type().into(), pointer.get_type().into()],
            false,
        );

        // Initialize the struct with size and pointer
        let mut list_value = list_type.const_zero(); // Start with a zeroed struct
        list_value = self
            .builder
            .build_insert_value(list_value, size, 0, "list_size")
            .expect("failed to inizlize struct")
            .into_struct_value();
        list_value = self
            .builder
            .build_insert_value(list_value, pointer, 1, "list_ptr")
            .expect("failed to initlize struct")
            .into_struct_value();

        // Return the new list as Value::List
        Ok(Value::List(List::Number(list_value)))
    }

    pub fn codegen_list_loop<F>(&self, lhs: List<'ctx>, func: F) -> Result<Value<'ctx>>
    where
        F: Fn(Number<'ctx>) -> Result<Number<'ctx>>, // The signature of the function
    {
        let size_field_index = 0; // Assuming size is at index 0
        let pointer_field_index = 1; // Assuming pointer is at index 1
                                     //
        match lhs {
            List::Number(lhs) => {
                // Get the size of the list
                let size_value = lhs
                    .get_field_at_index(size_field_index)
                    .context("Failed to get size field")?;
                let size: IntValue<'ctx> = size_value.into_int_value();

                // Get the pointer to the list elements
                let pointer_value: PointerValue<'ctx> = lhs
                    .get_field_at_index(pointer_field_index)
                    .context("Failed to get pointer field")?
                    .into_pointer_value();

                let current_fn = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                // Create a loop that iterates through the elements
                let entry_block = self.context.append_basic_block(current_fn, "entry");
                let loop_block = self.context.append_basic_block(current_fn, "loop");
                let end_block = self.context.append_basic_block(current_fn, "end");

                self.builder.position_at_end(entry_block);
                let index = self
                    .builder
                    .build_alloca(self.context.i64_type(), "index")?;
                self.builder
                    .build_store(index, self.context.i64_type().const_int(0, false));

                // Start the loop
                self.builder.build_unconditional_branch(loop_block);

                self.builder.position_at_end(loop_block);
                // Load index value
                let current_index =
                    self.builder
                        .build_load(self.context.i64_type(), index, "current_index")?;
                let current_index_value = current_index.into_int_value();

                // Check if we are still within bounds
                let condition = self.builder.build_int_compare(
                    inkwell::IntPredicate::ULT,
                    current_index_value,
                    size,
                    "condition",
                )?;

                // If condition fails, jump to end
                self.builder
                    .build_conditional_branch(condition, loop_block, end_block);

                // Calculate the pointer for the current element
                let element_ptr = unsafe {
                    self.builder.build_in_bounds_gep(
                        self.context.f64_type(),
                        pointer_value,
                        &[current_index_value],
                        "element_ptr",
                    )?
                };

                // Load the current element value
                let current_value = self.builder.build_load(
                    self.context.f64_type(),
                    element_ptr,
                    "current_value",
                )?;

                // Create a `Number` from the current value
                let lhs_value = Number::Float(current_value.into_float_value());

                // Call the provided function on the current value
                let new_value = func(lhs_value); // Invoke the function

                // Store the new value back in the list
                self.builder
                    .build_store(element_ptr, new_value?.as_basic_value_enum());

                // Increment the index
                let incremented_index = self.builder.build_int_add(
                    current_index_value,
                    self.context.i64_type().const_int(1, false),
                    "incremented_index",
                )?;
                self.builder.build_store(index, incremented_index);

                // Repeat the loop
                self.builder.build_unconditional_branch(loop_block);

                // End block
                self.builder.position_at_end(end_block);
                // Return the modified list
                Ok(Value::List(List::Number(lhs)))
            }
        }
    }

    pub fn codegen_binary_op(
        &self,
        lhs: Value<'ctx>,
        op: BinaryOp,
        rhs: Value<'ctx>,
    ) -> Result<Value<'ctx>> {
        match (lhs, rhs) {
            (Value::List(lhs), Value::Number(rhs)) => {
                self.codegen_list_loop(lhs, |lhs| self.codegen_binary_number_op(lhs, op, rhs))
            }
            (Value::Number(lhs), Value::Number(rhs)) => {
                Ok(Value::Number(self.codegen_binary_number_op(lhs, op, rhs)?))
            }
            (_, _) => Err(anyhow!(
                "typeerror, expected (List, Number) or (Number, Number)"
            )),
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

                    Number::from_basic_value_enum(
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
                            .try_as_basic_value()
                            .expect_left("return type should not be void"),
                    )
                    .expect("should be number type")
                }
                (Number::Int(lhs), Number::Int(rhs)) => {
                    let intrinsic = Intrinsic::find("llvm.powi").unwrap();

                    Number::from_basic_value_enum(
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
                            .try_as_basic_value()
                            .expect_left("return type should not be void"),
                    )
                    .expect("should be number type")
                }
                (Number::Int(lhs), Number::Float(rhs)) => {
                    let intrinsic = Intrinsic::find("llvm.pow").unwrap();

                    Number::from_basic_value_enum(
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
                            .try_as_basic_value()
                            .expect_left("return type should not be void"),
                    )
                    .expect("should be number type")
                }
                (Number::Float(lhs), Number::Int(rhs)) => {
                    let intrinsic = Intrinsic::find("llvm.powi").unwrap();

                    Number::from_basic_value_enum(
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
                            .try_as_basic_value()
                            .expect_left("return type should not be void"),
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

        let fn_type = match ret_type {
            ValueType::List(ListType::Number) => self
                .context
                .ptr_type(AddressSpace::default())
                .fn_type(&types, false),
            ValueType::Number(NumberType::Float) => self.context.f64_type().fn_type(&types, false),
            ValueType::Number(NumberType::Int) => self.context.i64_type().fn_type(&types, false),
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

    pub fn compile_all_exprs(&mut self) -> CompiledExprs<'ctx> {
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
