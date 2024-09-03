use anyhow::bail;
use inkwell::{
    types::{BasicMetadataTypeEnum, BasicType},
    values::{
        AnyValue, AnyValueEnum, BasicValue, BasicValueEnum, FloatValue, IntValue, PointerValue,
    },
    AddressSpace,
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

    pub fn from_any_value_enum(value: AnyValueEnum<'ctx>) -> Option<Self> {
        match value {
            AnyValueEnum::FloatValue(value) => Some(Self::Float(value)),
            AnyValueEnum::IntValue(value) => Some(Self::Int(value)),
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
pub enum Array<'ctx> {
    Number(PointerValue<'ctx>),
}

impl<'ctx> Array<'ctx> {
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

    pub fn get_type(&self) -> ArrayType {
        match self {
            Self::Number(_) => ArrayType::Number,
        }
    }

    pub fn from_pointer_value(value: PointerValue<'ctx>, ty: ArrayType) -> Self {
        match ty {
            ArrayType::Number => Array::Number(value),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ArrayType {
    Number,
}

#[derive(Debug, Clone, Copy)]
pub enum Value<'ctx> {
    Number(Number<'ctx>),
    Array(Array<'ctx>),
}

#[derive(Debug, Clone, Copy)]
pub enum ValueType {
    Number(NumberType),
    Array(ArrayType),
}

impl ValueType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Number(num_type) => num_type.name(),
        }
    }

    pub fn metadata<'ctx>(
        &self,
        context: &'ctx inkwell::context::Context,
    ) -> BasicMetadataTypeEnum<'ctx> {
        match self {
            Self::Number(number) => number.metadata(context),
            Self::Array(_) => BasicMetadataTypeEnum::StructType(
                context.struct_type(
                    &[
                        context
                            .ptr_type(AddressSpace::default())
                            .as_basic_type_enum(),
                        context.i64_type().as_basic_type_enum(),
                    ],
                    false,
                ),
            ),
        }
    }
}

impl<'ctx> Value<'ctx> {
    pub fn as_basic_value_enum(self) -> BasicValueEnum<'ctx> {
        match self {
            Value::Number(number) => number.as_basic_value_enum(),

            Value::Array(arr) => arr.as_basic_value_enum(),
        }
    }

    pub fn as_any_value_enum(self) -> AnyValueEnum<'ctx> {
        match self {
            Value::Number(number) => number.as_any_value_enum(),
            Value::Array(arr) => arr.as_any_value_enum(),
        }
    }

    pub fn get_type(&self) -> ValueType {
        match self {
            Value::Number(number) => ValueType::Number(number.get_type()),

            Value::Array(arr) => ValueType::Array(arr.get_type()),
        }
    }

    pub fn from_any_value_enum(value: AnyValueEnum<'ctx>, ty: ValueType) -> Option<Self> {
        match ty {
            ValueType::Number(_) => Number::from_any_value_enum(value).map(|n| Value::Number(n)),
            ValueType::Array(arr) => {
                if let AnyValueEnum::PointerValue(value) = value {
                    Some(Value::Array(Array::from_pointer_value(value, arr)))
                } else {
                    None
                }
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
