#![allow(unused)]
use std::io::{self, BufRead};

use anyhow::Result;
use parser::parse;

mod parser;

fn main() -> Result<()> {
    for line in io::stdin().lock().lines() {
        match parse(&line?) {
            Ok(expr) => {
                println!("{:?}", expr);
            }
            Err(e) => {
                eprintln!("Parse failed: {:?}", e);
            }
        }
    }
    Ok(())
}
