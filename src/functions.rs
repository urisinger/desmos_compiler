use crate::value::NumberType;

pub const IMPORTED_FUNCTIONS: [(&'static str, NumberType, &[NumberType], *const ()); 4] = [
    (
        "pow_f64_f64",
        NumberType::Float,
        &[NumberType::Float, NumberType::Float],
        pow_f64_f64 as *const (),
    ),
    (
        "pow_i64_i64",
        NumberType::Float,
        &[NumberType::Int, NumberType::Int],
        pow_i64_i64 as *const (),
    ),
    (
        "pow_i64_f64",
        NumberType::Float,
        &[NumberType::Int, NumberType::Float],
        pow_i64_f64 as *const (),
    ),
    (
        "pow_f64_i64",
        NumberType::Float,
        &[NumberType::Float, NumberType::Int],
        pow_f64_i64 as *const (),
    ),
];

#[no_mangle]
extern "C" fn pow_f64_f64(lhs: f64, rhs: f64) -> f64 {
    lhs.powf(rhs)
}

#[no_mangle]
extern "C" fn pow_f64_i64(lhs: f64, rhs: i64) -> f64 {
    lhs.powi(rhs as i32)
}

#[no_mangle]
extern "C" fn pow_i64_f64(lhs: i64, rhs: f64) -> f64 {
    (lhs as f64).powf(rhs)
}

#[no_mangle]
extern "C" fn pow_i64_i64(lhs: i64, rhs: i64) -> f64 {
    (lhs as f64).powi(rhs as i32)
}
