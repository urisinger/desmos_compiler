use inkwell::execution_engine::JitFunction;

use crate::lang::parser::ComparisonOp;

use self::jit::{ExplicitJitFn, ImplicitJitFn};

pub mod codegen;
pub mod jit;
pub mod types;
pub mod value;

pub struct CompiledExprs<'ctx> {
    pub compiled: Vec<CompiledExpr<'ctx>>,
}

pub enum CompiledExpr<'ctx> {
    Implicit {
        lhs: ImplicitJitFn<'ctx>,
        op: ComparisonOp,
        rhs: ImplicitJitFn<'ctx>,
    },
    Explicit {
        lhs: ExplicitJitFn<'ctx>,
    },
}
