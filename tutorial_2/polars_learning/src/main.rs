use polars::prelude::*;//{CsvReader, DataType, Field, Result as PolarResult, Schema, DataFrame,};
use polars::prelude::{Result as PolarResult};
use polars_core::prelude::*;
use polars::frame::DataFrame;
use std::fs::File;
use std::path::{Path};
// an eye on traits:
use polars::prelude::SerReader;
// If we don't import this one we might get the error CsvReader::new(file) new function not found in CsvReader

// What's Some?
/* Some is the same as haskell Just and Nothing, a variant enum 
   You can't use ? in the main function. ? is applied to Bar::version()? but it's implemented in main
   main return type is (), thus the compiler will complain because you can't use ? in main 
   due to its return type. So we need to implement the reading part in a function. 
   You could use fn main() -> Result<(), Error> {} using the nightly channel of Rust, not the stable

   underscore for unused variable

   Chuknked arrays: 
   Every Series contains a ChunkedArray<T>. Unlike Series, ChunkedArray’s are typed. 
   This allows us to apply closures to the data and collect the results to a ChunkedArray 
   of the same type T. Below we use an apply to use the cosine function to the values of a ChunkedArray.

   Excellent doc: https://docs.rs/polars/0.14.4/polars/chunked_array/struct.ChunkedArray.html

*/
pub fn read_csv<P: AsRef<Path>>(path: P) -> PolarResult<DataFrame> {
    /* Example function to create a dataframe from an input csv file*/
    let file = File::open(path).expect("Cannot open file.");

    CsvReader::new(file) 
    .has_header(true)
    .finish()
}

// do not return anything from this function
pub fn deal_with_shape<P: AsRef<Path>>(path: P) -> () {
    /* Example function to retrieve shape info from a dataframe */
    let df = read_csv(&path).unwrap();
    // shape 
    // reming {:#?} otherwise error ^^^^^ `(usize, usize)` cannot be formatted with the default formatter
    let shape = df.shape();
    println!("{:#?}", shape); 
    // schema
    println!("{:#?}", df.schema());
    // dtypes 
    println!("{:#?}", df.dtypes());
    // or width and height
    let width = df.width();
    println!("{}", width);
    let height = df.height();
    println!("{}", height);

}

// do not return anything from this function
pub fn deal_with_columns<P: AsRef<Path>>(path: P) -> () {
    /* Examples to deal with column and column names and enumerate */
    let df = read_csv(&path).unwrap();
    // column functions
    let columns = df.get_columns(); // you can do for column in columns{}
    let columname = df.get_column_names(); 

    // example like Python for i, val in enumerate(list, 0):
    for (i, column) in columns.iter().enumerate(){
        println!("{}, {}", column, columname[i]);
    }
}

// do not return anything from this function
pub fn deal_with_stacks<P: AsRef<Path>>(path: P) -> () {
    /* Stack, often happens to stack multiple dataframes together*/
    println!("Read the same dataframe twice");
    let df = read_csv(&path).unwrap();
    let df2 = read_csv(&path).unwrap();
    println!("Vertical stac the two dataframes");
    let mut df3 = df.vstack(&df2).unwrap(); // mut --> so we can change this dataframe later
    println!("{}, {:#?}", df3.head(Some(5)), df3.shape());
    // get column 
    println!("Get a column");
    let  sepal_length = df3.column("sepal.length").unwrap();
    println!("{}", sepal_length);
    println!("{:#?}", sepal_length.len());

    // drop columns
    println!("Drop a column");
    let sepal_length = df3.drop_in_place("sepal.length").unwrap(); // inplace
    // this commands return a Series
    println!("{}", df3.head(Some(5)));
    // drop_nulls() to drop NaN
    //let df4 = df3.drop("sepal.length"); // if we don't want a mut dataframe df3
    println!("Insert a series in a dataframe as a new column");
    let _df4 = df3.insert_at_idx(0, sepal_length).unwrap();
    println!("{}", _df4.head(Some(5)));

}

// Return a series
fn numb_to_log(in_df: &mut DataFrame) -> PolarResult<Series>{
    // do with a series  unwrap to have Series, .f64().uwrap() to retrieve a chunked array
    let to_log10_column = in_df.drop_in_place("sepal.length").unwrap().rename("log10.sepal.length")
                                                             .f64().unwrap() // create chunked array
                                                             .cast::<Float64Type>() // here we have apply
                                                             .unwrap() // unwrap because we have Result<>
                                                             .apply(|s| s.log10());
    
    let series10 = to_log10_column.into_series(); // reconvert into a series
    
    // return the column
    println!("{}", series10);
    Ok(series10)
}

pub fn deal_with_apply<P: AsRef<Path>>(path: P) -> () {
    /* Apply is one of the key functions in pandas*/
    let mut df = read_csv(&path).unwrap();
    // apply an operation or a function/closure 
    println!("Add 1 to first column");
    df.apply_at_idx(0, |s| s+1);
    println!("{}", df.head(Some(5)));
    // compute the log transform of a column and learn to play with series and chunked arrays
    let log10_series = numb_to_log(&mut df);
    // insert the column 
    println!(" log 10 of sepal length");
    df.with_column(log10_series.unwrap());
    println!("{}", df.head(Some(5)));
    // can we log transform throught apply_at_idx?
    df.apply_at_idx(0, |s| s.f64().unwrap().apply(|t| t.log10()));
    println!("{}", df.head(Some(5)));
}

fn main() {

    // here iris
    let ifile = "FULLPATHTO/iris.csv";
    // shape info 
    deal_with_shape(&ifile);
    // columns info 
    deal_with_columns(&ifile);
    // concatenate dataframe 
    deal_with_stacks(&ifile);
    // do math on this 
    deal_with_apply(&ifile);
    // train test split
    // nd array conversion, 2D only 
    let ifile = "FULLPATHTO/2d_array.csv";
    let df = read_csv(&ifile).unwrap();
    let ndarray = df.to_ndarray::<Float64Type>().unwrap();
    println!("{:?}", ndarray);
    
}
