use std::io::{self, BufRead, Write};

use anyhow::Result;
use expressions::Expressions;
use inkwell::context::Context;
use lang::backends::llvm::{CodeGen, CompiledExpr};

mod expressions;
mod lang;

fn main() -> Result<()> {
    let context = Context::create();

    let mut expressions = Expressions::new();
    loop {
        print!("> "); // Prompt the user
        io::stdout().flush().unwrap(); // Ensure the prompt is displayed immediately

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let line = input.trim();

        if line == "compile" {
            for (id, err) in &expressions.errors {
                println!("err: {err}");
            }
            let mut codegen = CodeGen::new(&context, &expressions);
            let compiled = codegen.compile_all_exprs();

            for expr in compiled.compiled {
                match expr {
                    CompiledExpr::Explicit { lhs } => {
                        let name = lhs.get_name().to_str()?;
                        let function = codegen.get_explicit_fn(name).unwrap();

                        let x = get_user_value();
                        unsafe { println!("function {name}({x}) returned {}", function.call(x)) };
                    }
                    CompiledExpr::Implicit { lhs, op, rhs } => {
                        let name = lhs.get_name().to_str()?;
                        let function = codegen.get_implicit_fn(name).unwrap();

                        let x = get_user_value();
                        let y = get_user_value();
                        unsafe {
                            println!("function {name}({x},{y}) returned {}", function.call(x, y))
                        };
                    }
                }
            }
        } else {
            expressions.add_expr(&line.to_string());
        }
    }
}

fn get_user_value() -> f64 {
    println!("Enter a value: ");

    let mut input_line = String::new();
    io::stdin()
        .read_line(&mut input_line)
        .expect("Failed to read line");

    input_line.trim().parse().expect("Input not an integer")
}
