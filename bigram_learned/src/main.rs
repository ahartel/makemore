use std::borrow::Cow;
use std::io::Write;
use std::{collections::HashMap, io};
use tch::{Device, Kind, Tensor};
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

const DEVICE: Device = Device::Cpu;

fn read_names() -> Vec<String> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(io::stdin());
    rdr.deserialize()
        .map(|r| {
            let s: String = r.unwrap();
            format!("..{s}.").to_string()
        })
        .collect()
}

#[derive(Clone)]
struct Stoi {
    _fwd: HashMap<char, i64>,
    _inv: HashMap<i64, char>,
}

impl Stoi {
    pub fn new() -> Self {
        let fwd = ".abcdefghijklmnopqrstuvwxyz"
            .chars()
            .zip(0..27)
            .collect::<HashMap<char, i64>>();
        let inv = (0..27)
            .zip(".abcdefghijklmnopqrstuvwxyz".chars())
            .collect::<HashMap<i64, char>>();
        Stoi {
            _fwd: fwd,
            _inv: inv,
        }
    }
    pub fn map(&self, c: char) -> i64 {
        self._fwd[&c]
    }
    pub fn inv(&self, idx: i64) -> char {
        self._inv[&idx]
    }
}

fn print_in_color(s: &str, saturation: u8) {
    let mut stdout = StandardStream::stdout(ColorChoice::Always);
    stdout
        .set_color(ColorSpec::new().set_fg(Some(Color::Rgb(saturation, saturation, saturation))))
        .unwrap();
    write!(&mut stdout, "{s}").unwrap();
    stdout.set_color(ColorSpec::new().set_fg(None)).unwrap();
}

#[derive(Debug)]
struct Error(Cow<'static, str>);

/// print a 2D tensor who's values will be interpreted as doubles
/// xlables and ylabels must be provided
/// relative magnitude of values will be indicated by grey values
fn print_tensor_2d(
    tensor: &Tensor,
    xlabels: Box<dyn Fn(i64) -> String>,
    ylabels: Box<dyn Fn(i64) -> String>,
) -> Result<(), Error> {
    if tensor.size().len() != 2 {
        return Err(Error("Tensor must have 2 dimensions".into()));
    }
    let max = tensor.max();
    let min = tensor.min();
    let size = tensor.size();
    for row in 0..size[0] {
        for col in 0..size[1] {
            let value = tensor.get(row).get(col);
            let saturation = (value - &min) / (&max - &min) * 256.0;
            let saturation = saturation.int64_value(&[]) as u8;
            print_in_color(&format!("{}{}    ", ylabels(row), xlabels(col)), saturation);
        }
        print!("\n");
        for col in 0..size[1] {
            let value = tensor.get(row).get(col);
            let saturation = (&value - &min) / (&max - &min) * 256.0;
            let saturation = saturation.int64_value(&[]) as u8;
            print_in_color(&format!("{:+.2} ", value.double_value(&[])), saturation);
        }
        print!("\n");
    }
    Ok(())
}

fn prepare_labels(stoi: &Stoi) -> (Box<dyn Fn(i64) -> String>, Box<dyn Fn(i64) -> String>) {
    let stoi_x = stoi.clone();
    let stoi_y = stoi.clone();
    let xlabels = Box::new(move |x| String::from(stoi_x.inv(x)));
    let ylabels = Box::new(move |y| String::from(stoi_y.inv(y)));
    (xlabels, ylabels)
}

fn calculate_loss(names: &[String], probs: &Tensor, stoi: &Stoi) -> Tensor {
    let mut total_log_likelihood = Tensor::zeros(&[1], (Kind::Float, DEVICE));
    let mut n = 0;
    for name in names {
        for (c1, c2) in name.chars().zip(name[1..].chars()) {
            let idx1 = stoi.map(c1);
            let idx2 = stoi.map(c2);
            n += 1;
            total_log_likelihood += probs.get(idx1).get(idx2).log();
        }
    }
    -total_log_likelihood / n
}

fn sample_words_bigram(num: usize, probs: &Tensor, stoi: &Stoi) {
    for _ in 0..num {
        let mut chars = Vec::new();
        let mut sampled = 0;
        loop {
            sampled = probs.get(sampled).multinomial(1, true).int64_value(&[]);
            chars.push(stoi.inv(sampled));
            if sampled == 0 {
                break;
            }
        }
        let name: String = chars.into_iter().collect();
        println!("{:?}", name);
    }
}

fn main() {
    let names = read_names();
    let stoi = Stoi::new();
    let mut w = Tensor::randn(&[27, 27], (Kind::Float, DEVICE)).set_requires_grad(true);
    let (xlabels, ylabels) = prepare_labels(&stoi);
    print_tensor_2d(&w, xlabels, ylabels).unwrap();
    let w_exp = w.exp();
    let row_sums = w_exp.sum_to_size(&[27, 1]);
    let probs = w_exp / row_sums;
    sample_words_bigram(15, &probs, &stoi);
    let loss = calculate_loss(&names, &probs, &stoi);
    println!("{}", loss.double_value(&[]));
    w.zero_grad();
    loss.backward();
    w.set_data(&(w.data() + w.grad()));
    let loss = calculate_loss(&names, &probs, &stoi);
    println!("{}", loss.double_value(&[]));
}
