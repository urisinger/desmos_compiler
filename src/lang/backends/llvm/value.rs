use anyhow::bail;
use inkwell::{
    types::{BasicMetadataTypeEnum, BasicType},
    values::{
        AnyValue, AnyValueEnum, BasicValue, BasicValueEnum, FloatValue, IntValue, PointerValue,
        StructValue,
    },
    AddressSpace,
};

use super::types::{ListType, NumberType, ValueType};

#[derive(Debug, Clone, Copy)]
pub enum Number<'ctx> {
    Float(FloatValue<'ctx>),
    Int(IntValue<'ctx>),
}

impl<'ctx> Number<'ctx> {
    pub fn as_basic_value_enum(self) -> BasicValueEnum<'ctx> {
        match self {
            Self::Float(value) => value.as_basic_value_enum(),
            Self::Int(value) => value.as_basic_value_enum(),
        }
    }

    pub fn as_any_value_enum(self) -> AnyValueEnum<'ctx> {
        match self {
            Self::Float(value) => value.as_any_value_enum(),
            Self::Int(value) => value.as_any_value_enum(),
        }
    }

    pub fn get_type(&self) -> NumberType {
        match self {
            Self::Float(_) => NumberType::Float,
            Self::Int(_) => NumberType::Int,
        }
    }

    pub fn from_basic_value_enum(value: BasicValueEnum<'ctx>) -> Option<Self> {
        match value {
            BasicValueEnum::FloatValue(value) => Some(Self::Float(value)),
            BasicValueEnum::IntValue(value) => Some(Self::Int(value)),
            _ => None,
        }
    }
}

impl<'ctx> From<FloatValue<'ctx>> for Number<'ctx> {
    fn from(value: FloatValue<'ctx>) -> Self {
        Self::Float(value)
    }
}

impl<'ctx> From<IntValue<'ctx>> for Number<'ctx> {
    fn from(value: IntValue<'ctx>) -> Self {
        Self::Int(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum List<'ctx> {
    Number(StructValue<'ctx>),
}

impl<'ctx> List<'ctx> {
    pub fn as_basic_value_enum(self) -> BasicValueEnum<'ctx> {
        match self {
            Self::Number(value) => value.as_basic_value_enum(),
        }
    }

    pub fn as_any_value_enum(self) -> AnyValueEnum<'ctx> {
        match self {
            Self::Number(value) => value.as_any_value_enum(),
        }
    }

    pub fn from_basic_value_enum(value: BasicValueEnum<'ctx>, ty: ListType) -> Option<Self> {
        match ty {
            ListType::Number => Some(List::Number(value.try_into().ok()?)),
        }
    }

    pub fn get_type(&self) -> ListType {
        match self {
            Self::Number(_) => ListType::Number,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Value<'ctx> {
    Number(Number<'ctx>),
    List(List<'ctx>),
}

impl<'ctx> Value<'ctx> {
    pub fn as_basic_value_enum(self) -> BasicValueEnum<'ctx> {
        match self {
            Value::Number(number) => number.as_basic_value_enum(),

            Value::List(arr) => arr.as_basic_value_enum(),
        }
    }

    pub fn as_any_value_enum(self) -> AnyValueEnum<'ctx> {
        match self {
            Value::Number(number) => number.as_any_value_enum(),
            Value::List(arr) => arr.as_any_value_enum(),
        }
    }

    pub fn get_type(&self) -> ValueType {
        match self {
            Value::Number(number) => ValueType::Number(number.get_type()),

            Value::List(arr) => ValueType::List(arr.get_type()),
        }
    }

    pub fn from_basic_value_enum(value: BasicValueEnum<'ctx>, ty: ValueType) -> Option<Self> {
        match ty {
            ValueType::Number(_) => Number::from_basic_value_enum(value).map(|n| Value::Number(n)),
            ValueType::List(arr_type) => {
                List::from_basic_value_enum(value, arr_type).map(|arr| Value::List(arr))
            }
        }
    }
}

impl<'ctx> From<FloatValue<'ctx>> for Value<'ctx> {
    fn from(value: FloatValue<'ctx>) -> Self {
        Self::Number(Number::Float(value))
    }
}

impl<'ctx> From<IntValue<'ctx>> for Value<'ctx> {
    fn from(value: IntValue<'ctx>) -> Self {
        Self::Number(Number::Int(value))
    }
}
