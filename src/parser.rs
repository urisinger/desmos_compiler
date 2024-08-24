use std::iter;

use anyhow::Result;
use lazy_static::lazy_static;
use pest::iterators::Pairs;

use pest::pratt_parser::{Assoc, Op, PrattParser};
use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "desmos.pest"] // Point to the grammar file
struct ExprParser;

lazy_static! {
    static ref PRATT_PARSER: PrattParser<Rule> = {
        PrattParser::new()
            .op(Op::infix(Rule::add, Assoc::Left) | Op::infix(Rule::sub, Assoc::Left))
            .op(Op::infix(Rule::mul, Assoc::Left)
                | Op::infix(Rule::div, Assoc::Left)
                | Op::infix(Rule::paren, Assoc::Left))
            .op(Op::postfix(Rule::fac) | Op::postfix(Rule::pow))
            .op(Op::prefix(Rule::neg)
                | Op::prefix(Rule::sqrt)
                | Op::prefix(Rule::sin)
                | Op::prefix(Rule::cos)
                | Op::prefix(Rule::tan)
                | Op::prefix(Rule::csc)
                | Op::prefix(Rule::sec)
                | Op::prefix(Rule::cot)
                | Op::prefix(Rule::invsin)
                | Op::prefix(Rule::invcos)
                | Op::prefix(Rule::invtan)
                | Op::prefix(Rule::invcsc)
                | Op::prefix(Rule::invsec)
                | Op::prefix(Rule::invcot))
    };
}

#[derive(Debug)]
pub enum Expr {
    Int(i32),
    Float(f64),
    Ident(String),
    BinOp {
        lhs: Box<Expr>,
        op: BinaryOp,
        rhs: Box<Expr>,
    },
    UnaryOp {
        val: Box<Expr>,
        op: UnaryOp,
    },
    FnCall {
        ident: String,
        args: Vec<Box<Expr>>,
    },
    Tuple {
        args: Vec<Box<Expr>>,
    },
}

#[derive(Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug)]
pub enum UnaryOp {
    Neg,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Csc,
    Sec,
    Cot,
    InvSin,
    InvCos,
    InvTan,
    InvCsc,
    InvSec,
    InvCot,
    Fac,
}

pub fn parse(s: &str) -> Result<Expr> {
    let pairs = ExprParser::parse(Rule::program, s)?
        .next()
        .unwrap()
        .into_inner();
    parse_expr(pairs)
}

fn parse_expr(pairs: Pairs<Rule>) -> Result<Expr> {
    PRATT_PARSER
        .map_primary(|primary| {
            Ok(match primary.as_rule() {
                Rule::int => Expr::Int(primary.as_str().parse::<i32>()?),
                Rule::ident => Expr::Ident(primary.as_str().to_string()),
                Rule::expr => parse_expr(primary.into_inner())?,
                Rule::fn_call => {
                    let mut pairs = primary.into_inner();
                    let func_name = pairs.next().unwrap();
                    match func_name.as_rule() {
                        Rule::ident => {
                            let args = pairs
                                .map(|expr| parse_expr(expr.into_inner()).map(Box::new))
                                .collect::<Result<Vec<_>>>()?;

                            Expr::FnCall {
                                args,
                                ident: func_name.as_str().to_string(),
                            }
                        }
                        _ => {
                            if pairs.len() == 0 {
                                parse_expr(func_name.into_inner())?
                            } else {
                                let args = iter::once(func_name)
                                    .chain(pairs)
                                    .map(|expr| parse_expr(expr.into_inner()).map(Box::new))
                                    .collect::<Result<Vec<_>>>()?;

                                Expr::Tuple { args }
                            }
                        }
                    }
                }
                Rule::float => Expr::Float(primary.as_str().parse::<f64>()?),
                Rule::frac => {
                    let mut pairs = primary.into_inner();
                    let numerator = pairs.next().unwrap();
                    let denominator = pairs.next().unwrap();
                    Expr::BinOp {
                        lhs: Box::new(parse_expr(numerator.into_inner())?),
                        op: BinaryOp::Div,
                        rhs: Box::new(parse_expr(denominator.into_inner())?),
                    }
                }
                rule => unreachable!(
                    "Expr::parse expected fn_call, int, expr, ident, found {:?}",
                    rule
                ),
            })
        })
        .map_prefix(|op, rhs| {
            Ok(match op.as_rule() {
                Rule::neg => Expr::UnaryOp {
                    val: Box::new(rhs?),
                    op: UnaryOp::Neg,
                },

                rule => Expr::UnaryOp {
                    val: Box::new(rhs?),
                    op: match rule {
                        Rule::neg => UnaryOp::Neg,
                        Rule::sqrt => UnaryOp::Sqrt,
                        Rule::sin => UnaryOp::Sin,
                        Rule::cos => UnaryOp::Cos,
                        Rule::tan => UnaryOp::Tan,
                        Rule::csc => UnaryOp::Csc,
                        Rule::sec => UnaryOp::Sec,
                        Rule::cot => UnaryOp::Cot,
                        Rule::invsin => UnaryOp::InvSin,
                        Rule::invcos => UnaryOp::InvCos,
                        Rule::invtan => UnaryOp::InvTan,
                        Rule::invcsc => UnaryOp::InvCsc,
                        Rule::invsec => UnaryOp::InvSec,
                        Rule::invcot => UnaryOp::InvCot,
                        _ => unreachable!(),
                    },
                },
            })
        })
        .map_postfix(|lhs, op| {
            Ok(match op.as_rule() {
                Rule::pow => {
                    let mut pairs = op.into_inner();
                    let exponent = pairs.next().unwrap();
                    Expr::BinOp {
                        lhs: Box::new(lhs?),
                        op: BinaryOp::Pow,
                        rhs: Box::new(parse_expr(exponent.into_inner())?),
                    }
                }

                _ => unreachable!(),
            })
        })
        .map_infix(|lhs, op, rhs| {
            Ok(Expr::BinOp {
                lhs: Box::new(lhs?),
                op: match op.as_rule() {
                    Rule::add => BinaryOp::Add,
                    Rule::sub => BinaryOp::Sub,
                    Rule::mul => BinaryOp::Mul,
                    Rule::div => BinaryOp::Div,
                    Rule::pow => BinaryOp::Pow,
                    Rule::paren => BinaryOp::Mul,
                    _ => unreachable!(),
                },
                rhs: Box::new(rhs?),
            })
        })
        .parse(pairs)
}
