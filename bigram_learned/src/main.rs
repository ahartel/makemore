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
            format!(".{s}.").to_string()
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

fn calculate_loss(names: &[String], w: &Tensor, stoi: &Stoi) -> Tensor {
    let mut xs: Vec<i64> = Vec::new();
    let mut ys: Vec<i64> = Vec::new();
    for name in names {
        for (c1, c2) in name.chars().zip(name[1..].chars()) {
            let idx1 = stoi.map(c1);
            let idx2 = stoi.map(c2);
            xs.push(idx1);
            ys.push(idx2);
        }
    }
    let xs = Tensor::of_slice(&xs);
    let ys = Tensor::of_slice(&ys).to_dtype(Kind::Int64, false, false);
    let xenc = xs.onehot(27);
    let num = xenc.size()[0];
    let logits = xenc.matmul(w);
    let counts = logits.exp();
    let sum = counts.sum_to_size(&[num, 1]);
    let probs = &counts / sum;
    let loss = -probs
        .index(&[Some(Tensor::arange(num, (Kind::Int64, DEVICE))), Some(ys)])
        .log()
        .mean(Kind::Float);
    loss
}

fn sample_words_bigram(num: usize, w: &Tensor, stoi: &Stoi) {
    for _ in 0..num {
        let mut chars = Vec::new();
        let mut sampled = Tensor::of_slice(&[0 as i32]);
        loop {
            let x = sampled.onehot(27);
            let logits = x.matmul(w);
            let probs = logits.exp() / logits.exp().sum_to_size(&[1]);
            sampled = probs.multinomial(1, true).squeeze();
            let sampled_int_value = sampled.int64_value(&[]);
            chars.push(stoi.inv(sampled_int_value));
            if sampled_int_value == 0 {
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
    sample_words_bigram(15, &w, &stoi);
    for i in 0..100 {
        let loss = calculate_loss(&names, &w, &stoi);
        println!("Loss after {} iterations: {}", i, loss.double_value(&[]));
        w.zero_grad();
        loss.backward();
        w.set_data(&(w.data() - 10.0 * w.grad()));
    }
    sample_words_bigram(15, &w, &stoi);
}
