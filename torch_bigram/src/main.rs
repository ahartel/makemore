use std::io::Write;
use std::{collections::HashMap, io};
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

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
struct Stoi {
    fwd: HashMap<char, usize>,
    inv: HashMap<usize, char>,
}

impl Stoi {
    pub fn new() -> Self {
        let fwd = ".abcdefghijklmnopqrstuvwxyz"
            .chars()
            .zip(0..27)
            .collect::<HashMap<char, usize>>();
        let inv = ".abcdefghijklmnopqrstuvwxyz"
            .chars()
            .enumerate()
            .collect::<HashMap<usize, char>>();
        Stoi { fwd, inv }
    }
    pub fn map(&self, c: char) -> usize {
        self.fwd[&c]
    }
    pub fn inv(&self, idx: usize) -> char {
        self.inv[&idx]
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

fn print_bigrams(bigrams: &Vec<Vec<usize>>, stoi: &Stoi) {
    let max = *bigrams
        .iter()
        .map(|col| col.iter().max().unwrap())
        .max()
        .unwrap() as f32;
    for row in 0..27 {
        for col in 0..27 {
            let saturation = (bigrams[row][col] as f32 / max * 256.0) as u8;
            print_in_color(
                &format!("  {}{} ", stoi.inv(row), stoi.inv(col)),
                saturation,
            );
        }
        print!("\n");
        for col in 0..27 {
            let saturation = (bigrams[row][col] as f32 / max * 256.0) as u8;
            print_in_color(&format!("{:0>4} ", bigrams[row][col]), saturation);
        }
        print!("\n");
    }
}

fn main() {
    let names = read_names();
    let stoi = Stoi::new();
    // let mut bigrams = Tensor::f_zeros(&[27, 27], (Kind::Float, Device::Cpu)).unwrap();
    let mut bigrams: Vec<Vec<usize>> = vec![vec![0; 27]; 27];
    for word in names.iter() {
        for (c1, c2) in word.chars().zip(word[1..].chars()) {
            let idx1 = stoi.map(c1);
            let idx2 = stoi.map(c2);
            bigrams[idx1][idx2] += 1;
        }
    }
    print_bigrams(&bigrams, &stoi);
}
