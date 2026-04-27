use anyhow::Result;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Merge two JSON values according to these rules:
/// - If either value is null, prefer the non-null value
/// - For primitive types (bool, number, string), right overrides left unless special case
/// - Empty strings on the right preserve the left value
/// - Arrays are replaced by the right value
/// - Objects are deeply merged recursively
fn merge_json(left: Value, right: Value) -> Value {
    match (left, right) {
        (Value::Null, right) => right,
        (left, Value::Null) => left,
        (Value::Bool(_), right) => right,
        (Value::Number(_), right) => right,
        (left @ Value::String(_), Value::String(s)) if s == "" => left,
        (Value::String(_), right) => right,
        (Value::Array(_), right) => right,
        (Value::Object(mut left), Value::Object(right)) => {
            for (key, right_value) in right {
                let new_value = match left.remove(&key) {
                    Some(left_value) => merge_json(left_value, right_value),
                    None => right_value,
                };
                left.insert(key, new_value);
            }
            Value::Object(left)
        }
        (Value::Object(_), right) => right,
    }
}

pub fn merge<Left, Right, Output>(left: Left, right: Right) -> Result<Output>
where
    Left: Serialize,
    Right: Serialize,
    Output: DeserializeOwned,
{
    Ok(serde_json::from_value(merge_json(
        serde_json::to_value(left)?,
        serde_json::to_value(right)?,
    ))?)
}
