use anyhow::bail;
use inkwell::{
    types::BasicMetadataTypeEnum,
    values::{AnyValue, AnyValueEnum, BasicValue, BasicValueEnum, FloatValue, IntValue},
};

#[derive(Debug, Clone, Copy)]
pub enum NumberType {
    Float,
    Int,
}

impl NumberType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Int => "int",
        }
    }

    pub fn metadata<'ctx>(
        &self,
        context: &'ctx inkwell::context::Context,
    ) -> BasicMetadataTypeEnum<'ctx> {
        match self {
            Self::Float => BasicMetadataTypeEnum::FloatType(context.f64_type()),
            Self::Int => BasicMetadataTypeEnum::IntType(context.i64_type()),
        }
    }
}

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

impl<'ctx> TryFrom<AnyValueEnum<'ctx>> for Number<'ctx> {
    type Error = anyhow::Error;
    fn try_from(value: AnyValueEnum<'ctx>) -> Result<Self, Self::Error> {
        match value {
            AnyValueEnum::FloatValue(value) => Ok(Self::Float(value)),
            AnyValueEnum::IntValue(value) => Ok(Self::Int(value)),
            val => bail!("expected float or int value, found {}", val.get_type()),
        }
    }
}

impl<'ctx> Into<BasicValueEnum<'ctx>> for Number<'ctx> {
    fn into(self) -> BasicValueEnum<'ctx> {
        self.as_basic_value_enum()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Value<'ctx> {
    Number(Number<'ctx>),
}

#[derive(Debug, Clone, Copy)]
pub enum ValueType {
    Number(NumberType),
}

impl ValueType {
    pub fn name(&self) -> &'static str {
        match self {
            ValueType::Number(num_type) => num_type.name(),
        }
    }

    pub fn metadata<'ctx>(
        &self,
        context: &'ctx inkwell::context::Context,
    ) -> BasicMetadataTypeEnum<'ctx> {
        match self {
            Self::Number(number) => number.metadata(context),
        }
    }
}

impl<'ctx> Value<'ctx> {
    pub fn as_basic_value_enum(self) -> BasicValueEnum<'ctx> {
        match self {
            Value::Number(number) => number.as_basic_value_enum(),
        }
    }

    pub fn as_any_value_enum(self) -> AnyValueEnum<'ctx> {
        match self {
            Value::Number(number) => number.as_any_value_enum(),
        }
    }

    pub fn get_type(&self) -> ValueType {
        match self {
            Value::Number(number) => ValueType::Number(number.get_type()),
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

impl<'ctx> TryFrom<AnyValueEnum<'ctx>> for Value<'ctx> {
    type Error = anyhow::Error;
    fn try_from(value: AnyValueEnum<'ctx>) -> Result<Self, Self::Error> {
        match value {
            AnyValueEnum::FloatValue(value) => Ok(Self::Number(Number::Float(value))),
            AnyValueEnum::IntValue(value) => Ok(Self::Number(Number::Int(value))),
            val => bail!("expected float or int value, found {}", val.get_type()),
        }
    }
}
