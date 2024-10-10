use inkwell::{
    types::{BasicMetadataTypeEnum, BasicType},
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
pub enum ListType {
    Number,
}

impl ListType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Number => "number_list",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ValueType {
    Number(NumberType),
    List(ListType),
}

impl ValueType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Number(num_type) => num_type.name(),
            Self::List(list_type) => list_type.name(),
        }
    }

    pub fn metadata<'ctx>(
        &self,
        context: &'ctx inkwell::context::Context,
    ) -> BasicMetadataTypeEnum<'ctx> {
        match self {
            Self::Number(number) => number.metadata(context),
            Self::List(_) => BasicMetadataTypeEnum::StructType(
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
