extern crate serde;
#[macro_use]
extern crate serde_derive; 

use std::vec::Vec; 
use std::env::args; 

mod linear_regression;

fn main() {
    println!("Hello, world!");
    linear_regression::run();
}
