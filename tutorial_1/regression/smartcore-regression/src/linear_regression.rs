// usual imports 
use std::error::Error;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::fs::File;
use std::vec::Vec;
// use the TryFrom to convert exactly the nubmer 
use std::convert::TryFrom;

// smartcore 
use smartcore::linear::linear_regression::LinearRegression; 
// matrix format 
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// metrics 
use smartcore::metrics::mean_squared_error;
// traintest split
use smartcore::model_selection::train_test_split;

// import the reader 
use read_csv::read_housing_csv;


// create a function for using linear regression 
pub fn run() -> f64 {

    // Read input data 
    let ifile = "FULLPATHTODATASET";
    let mut input_data = read_housing_csv(&ifile);
    let input_data_size = input_data.len();
    let n_cols = usize::try_from(13).unwrap();
    println!("n_cols {:?}", n_cols);

    // create X and y
    let X: Vec<f64> = input_data.iter().flat_map(|row| row.train_features()).collect(); 
    let y: Vec<f64> = input_data.iter().map(|row| row.train_target()).collect();
    // to matrix 
    let Xmatrix = DenseMatrix::from_array(input_data_size, n_cols, &X); // size of the array, numb of features 
    // split
    let (x_train, x_test, y_train, y_test) = train_test_split(&Xmatrix, &y, 0.3, true);
    // model 
    let linear_regression = LinearRegression::fit(&x_train, &y_train, Default::default()).unwrap();
    // predictions 
    let preds = linear_regression.predict(&x_test).unwrap(); 
    // metrics 
    let mse = mean_squared_error(&y_test, &preds);
    println!("MSE: {:?}", mse); 
    mse
}