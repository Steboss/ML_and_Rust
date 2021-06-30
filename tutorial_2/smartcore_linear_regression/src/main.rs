// usual imports 
use std::path::Path;
use std::fs::File;
use std::vec::Vec;
// use the TryFrom to convert exactly the nubmer 
use std::convert::TryFrom;

// smartcore 
use smartcore::linear::linear_regression::LinearRegression; 
// matrix format 
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;
// metrics 
use smartcore::metrics::mean_squared_error;
// traintest split
use smartcore::model_selection::train_test_split;

// polars
use polars::prelude::*;//{CsvReader, DataType, Field, Result as PolarResult, Schema, DataFrame,};
use polars::prelude::{Result as PolarResult};
use polars::frame::DataFrame;
use polars::prelude::SerReader;



pub fn read_csv<P: AsRef<Path>>(path: P) -> PolarResult<DataFrame> {
    /* Example function to create a dataframe from an input csv file*/
    let file = File::open(path).expect("Cannot open file.");

    CsvReader::new(file)
    .has_header(true)
    //.with_delimiter(b' ')
    //NB b" " throws an error
    .finish()
}

pub fn feature_and_target(in_df: &DataFrame) -> (PolarResult<DataFrame>,PolarResult<DataFrame>) {
    /* Read a dataframe, select the columns we need for feature training and target and return 
    the two new dataframes*/
    let features = in_df.select(vec!["crim",
                                "zn",
                                "indus",
                                "chas",
                                "nox",
                                "rm",
                                "age",
                                "dis",
                                "rad",
                                "tax",
                                "ptratio",
                                "black",
                                "lstat"]);

    let target = in_df.select("medv");

    (features, target)
    
}

pub fn convert_features_to_matrix(in_df: &DataFrame) -> Result<DenseMatrix<f64>>{
    /* function to convert feature dataframe to a DenseMatrix, readable by smartcore*/

    let nrows = in_df.height();
    let ncols = in_df.width();
    // convert to array
    let features_res = in_df.to_ndarray::<Float64Type>().unwrap();
    // create a zero matrix and populate with features
    let mut Xmatrix: DenseMatrix<f64> = BaseMatrix::zeros(nrows, ncols);
    // populate the matrix 
    // initialize row and column counters
    let mut col:  u32 = 0;
    let mut row:  u32 = 0;

    for val in features_res.iter(){
        
        // Debug
        //println!("{},{}", usize::try_from(row).unwrap(), usize::try_from(col).unwrap());
        // define the row and col in the final matrix as usize
        let m_row = usize::try_from(row).unwrap();
        let m_col = usize::try_from(col).unwrap();
        // NB we are dereferencing the borrow with *val otherwise we would have a &val type, which is 
        // not what set wants
        xmatrix.set(m_row, m_col, *val);
        // check what we have to update
        if (m_col==ncols-1) {
            row+=1;
            col = 0;
        } else{
            col+=1;
        }
    }

    // Ok so we can return DenseMatrix, otherwise we'll have std::result::Result<Densematrix, PolarsError>
    Ok(xmatrix)
}

// create a function for using linear regression 
fn main() {

    // Read input data 
    let ifile = "FULLPATHTO/boston_dataset.csv";
    let df = read_csv(&ifile).unwrap();
    // select features
    let (features, target) = feature_and_target(&df);
    let xmatrix = convert_features_to_matrix(&features.unwrap());
    /* Example for runnign the function in main, note the clone() 
    // size and shape of features
    // clone to avoid error move occurs because feature has type result<> which does not implement the copy trait
    let features_clone = features.unwrap().clone();
    let nrows = features_clone.height();
    let ncols = features_clone.width();
    //println!("{:#?}", features.unwrap());
    // convert to array
    let features_res = features_clone.to_ndarray::<Float64Type>().unwrap();
    */
    let target_array = target.unwrap().to_ndarray::<Float64Type>().unwrap();
    // create a vec type and populate with y values
    let mut y: Vec<f64> = Vec::new();
    for val in target_array.iter(){
        y.push(*val);
    }
    /*
    // create a zero matrix and populate with features
    let mut Xmatrix: DenseMatrix<f64> = BaseMatrix::zeros(nrows, ncols);
    // populate the matrix 
    // initialize row and column counters
    let mut col:  u32 = 0;
    let mut row:  u32 = 0;

    for val in features_res.iter(){
        
        // Debug
        //println!("{},{}", usize::try_from(row).unwrap(), usize::try_from(col).unwrap());
        // define the row and col in the final matrix as usize
        let m_row = usize::try_from(row).unwrap();
        let m_col = usize::try_from(col).unwrap();
        // NB we are dereferencing the borrow with *val otherwise we would have a &val type, which is 
        // not what set wants
        Xmatrix.set(m_row, m_col, *val);
        // check what we have to update
        if (m_col==ncols-1) {
            row+=1;
            col = 0;
        } else{
            col+=1;
        }
    }
    */
    // train split 
    let (x_train, x_test, y_train, y_test) = train_test_split(&xmatrix.unwrap(), &y, 0.3, true);

    // model 
    let linear_regression = LinearRegression::fit(&x_train, &y_train, Default::default()).unwrap();
    // predictions 
    let preds = linear_regression.predict(&x_test).unwrap(); 
    // metrics 
    let mse = mean_squared_error(&y_test, &preds);
    println!("MSE: {:?}", mse);
}