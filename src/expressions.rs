use anyhow::{Context, Result};

use crate::lang::parser::{ComparisonOp, Expr, Node};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ExpressionId(pub u32);

impl From<u32> for ExpressionId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Debug)]
pub struct Expressions {
    pub exprs: HashMap<ExpressionId, Expr>,
    pub errors: HashMap<ExpressionId, String>,
    idents: HashMap<String, ExpressionId>,

    max_id: u32,
}

impl Expressions {
    pub fn new() -> Self {
        Self {
            exprs: Default::default(),
            idents: Default::default(),
            errors: Default::default(),
            max_id: 0,
        }
    }

    pub fn remove_expr(&mut self, id: impl Into<ExpressionId>) {
        let k = id.into();
        if let Some(expr) = self.exprs.remove(&k) {
            match &expr {
                Expr::VarDef { ident, .. } | Expr::FnDef { ident, .. } => {
                    self.idents.remove(ident);
                }
                _ => (),
            };
        }
    }

    pub fn add_expr(&mut self, s: &str) {
        self.set_expr(self.max_id, s);
        self.max_id += 1;
    }

    pub fn set_expr(&mut self, id: impl Into<ExpressionId>, s: &str) {
        let k = id.into();
        let expr = self.parse_expr(&s, k);
        match expr {
            Ok(expr) => {
                dbg!(&expr);
                self.exprs.insert(k, expr);
            }
            Err(e) => {
                self.errors.insert(k, e.to_string());
            }
        }
    }

    pub fn get_expr(&self, name: &str) -> Option<&Expr> {
        if let Some(id) = self.idents.get(name) {
            self.exprs.get(id)
        } else {
            None
        }
    }

    fn parse_expr(&mut self, s: &str, k: ExpressionId) -> Result<Expr> {
        let expr = Expr::parse(&s)?;
        match &expr {
            Expr::VarDef { ident, .. } | Expr::FnDef { ident, .. } => {
                self.idents.insert(ident.clone(), k);
            }
            _ => (),
        };

        Ok(expr)
    }
}
