use std::io::{self, BufRead};

use anyhow::Result;
use codegen::CodeGen;
use expressions::Expressions;
use inkwell::context::Context;

mod codegen;
mod expressions;
mod parser;
mod value;

fn main() -> Result<()> {
    let context = Context::create();

    let mut expressions = Expressions::new();
    for line in io::stdin().lock().lines() {
        let mut codegen = CodeGen::new(&context, &mut expressions);
        let line = line?;
        if &line == "compile" {
            codegen.compile_all_exprs();
            for (_, err) in &expressions.errors {
                eprintln!("{}", err);
            }
        } else {
            expressions.add_expr(line);
        }
    }
    Ok(())
}
