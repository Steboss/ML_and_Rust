// usual imports 
use std::error::Error;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::fs::File;
use std::vec::Vec;


// import rusty_machine for linear regression 
// as well as rust_machine Matrix and Vector as we need them for input 
use rusty_machine; 
use rusty_machine::linalg::Matrix; // surprise, Matrix is in linear algebra
use rusty_machine::linalg::BaseMatrix; // implement traits for matrix
use rusty_machine::linalg::Vector; 
use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::learning::SupModel; // this is extremelly important
// as this implement the traits to user predict and train
//use smartcore::metrics::*;
use rusty_machine::analysis::score::neg_mean_squared_error;
// import the reader 
use read_csv::read_housing_csv;




// create a function for using linear regression 
pub fn run() -> f64 {

    // Read input data 
    let ifile = "FULLPATHTODATASET";
    let mut input_data = read_housing_csv(&ifile);

    // data preprocessing: train test split 
    // define the length of the test data 
    let test_chunk_size: f64 = input_data.len() as f64 * 0.3; // we are taking the 30% as test data
    // as cast between types 
    let test_chunk_size = test_chunk_size.round() as usize;
    // split 
    let (test, train) = input_data.split_at(test_chunk_size); 
    // impressively, rust vectors have split_at attribute
    //https://doc.rust-lang.org/std/primitive.slice.html#method.split_at
    let train_size = train.len() ; 
    let test_size = test.len();
    // define the training and target variables 
    // to return train and target variables use a simple flat map 
    let x_train: Vec<f64> = train.iter().flat_map(|row| row.train_features()).collect();
    let y_train: Vec<f64> = train.iter().map(|row| row.train_target()).collect();
    // same for test
    let x_test: Vec<f64> = test.iter().flat_map(|row| row.train_features()).collect();
    let y_test: Vec<f64> = test.iter().map(|row| row.train_target()).collect();


    // now as an input linregressionwants a matrix and a vector for y so 
    let x_train_matrix = Matrix::new(train_size, 13, x_train); // 13 is the number of features
    let y_train_vector = Vector::new(y_train);
    let x_test_matrix = Matrix::new(test_size, 13, x_test);

    // MODEL! 
    let mut linearRegression = LinRegressor::default(); 
    // train 
    linearRegression.train(&x_train_matrix, &y_train_vector);
    // predictions
    let preds = linearRegression.predict(&x_test_matrix).unwrap();
    // convert to matrix both preds and y_test 
    let preds_matrix = Matrix::new(test_size, 1, preds); 
    let y_preds_matrix = Matrix::new(test_size, 1, y_test);
    // compute the mse
    let mse = neg_mean_squared_error(&preds_matrix, &y_preds_matrix); 

    println!("Final negMSE (the higher the better) {:?}", mse);
    // return the mse 
    mse
// further example with smartcore
}