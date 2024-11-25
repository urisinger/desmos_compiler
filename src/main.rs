#![allow(dead_code)]
use std::io::{self, Write};

use anyhow::Result;
use expressions::Expressions;
use inkwell::context::Context;
use lang::backends::llvm::{jit::ListLayout, CompiledExpr};

use crate::lang::backends::llvm::{
    codegen::compile_all_exprs,
    jit::{ExplicitJitFn, ExplicitJitListFn, ImplicitJitFn, ImplicitJitListFn},
};

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
            let compiled = compile_all_exprs(&context, &expressions);

            for expr in compiled.compiled {
                match expr {
                    CompiledExpr::Explicit { lhs } => {
                        let x = get_user_value();

                        match lhs {
                            ExplicitJitFn::Number(lhs) => {
                                println!("call to explicit with {x} returned {}", unsafe {
                                    lhs.call(x)
                                });
                            }
                            ExplicitJitFn::List(lhs) => match lhs {
                                ExplicitJitListFn::Number(lhs) => {
                                    let result = unsafe { lhs.call(x) };
                                    unsafe { print_list("explicit", result) };
                                }
                            },
                        }
                    }
                    CompiledExpr::Implicit { lhs, rhs, .. } => {
                        println!("got implicit");
                        let x = get_user_value();
                        let y = get_user_value();
                        print!("call to implicit with {x},{y} returned ");

                        match lhs {
                            ImplicitJitFn::Number(lhs) => {
                                println!("({}", unsafe { lhs.call(x, y) });
                            }
                            ImplicitJitFn::List(lhs) => match lhs {
                                ImplicitJitListFn::Number(lhs) => {
                                    let result = unsafe { lhs.call(x, y) };
                                    unsafe { print_list("implicit lhs", result) };
                                }
                            },
                        };

                        match rhs {
                            ImplicitJitFn::Number(rhs) => {
                                println!("{})", unsafe { rhs.call(x, y) });
                            }
                            ImplicitJitFn::List(rhs) => match rhs {
                                ImplicitJitListFn::Number(rhs) => {
                                    let result = unsafe { rhs.call(x, y) };
                                    unsafe { print_list("implicit rhs", result) };
                                }
                            },
                        };
                    }
                }
            }
        } else {
            _ = expressions.add_expr(line).inspect_err(|err| {
                println!("err: {err}");
            });
        }
    }
}

unsafe fn print_list(context: &str, list: ListLayout) {
    if list.size == 0 {
        println!("{context} list is empty.");
        return;
    }

    let ptr = list.ptr as *const f64; // Interpret the pointer as f64 pointer
    let size = list.size as usize;

    print!("{context} list: [");
    for i in 0..size {
        if i > 0 {
            print!(", ");
        }
        let value = *ptr.add(i); // Dereference the pointer with an offset
        print!("{value}");
    }
    println!("]");
}

fn get_user_value() -> f64 {
    println!("Enter a value: ");

    let mut input_line = String::new();
    io::stdin()
        .read_line(&mut input_line)
        .expect("Failed to read line");

    input_line.trim().parse().expect("Input not an integer")
}
