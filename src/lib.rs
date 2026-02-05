use std::collections::HashMap;
use std::io::Write;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    /// ゼロ初期化されたテンソルを生成
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor {
            shape,
            data: vec![0.0; size],
        }
    }

    /// 2次元ベクタからテンソルを生成
    pub fn from_vec2(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = data[0].len(); // 各行の長さは等しいと仮定
        let shape = vec![rows, cols];
        let mut flat_data = Vec::with_capacity(rows * cols);
        for row in data {
            assert_eq!(row.len(), cols, "行の長さが不一致です");
            flat_data.extend(row);
        }
        Tensor { shape, data: flat_data }
    }

    /// 3次元ベクタからテンソルを生成
    pub fn from_vec3(data: Vec<Vec<Vec<f64>>>) -> Self {
        let d0 = data.len();
        let d1 = data[0].len();
        let d2 = data[0][0].len();
        let shape = vec![d0, d1, d2];
        let mut flat_data = Vec::with_capacity(d0 * d1 * d2);
        
        for i in 0..d0 {
            assert_eq!(data[i].len(), d1, "次元1の長さが不一致です at index {}", i);
            for j in 0..d1 {
                assert_eq!(data[i][j].len(), d2, "次元2の長さが不一致です at index [{}, {}]", i, j);
                flat_data.extend(&data[i][j]);
            }
        }
        Tensor { shape, data: flat_data }
    }

    /// インデックスで要素を取得 (読み取り専用)
    pub fn get(&self, indices: &[usize]) -> f64 {
        let flat_index = self.calculate_flat_index(indices);
        self.data[flat_index]
    }

    /// インデックスで要素を設定
    pub fn set(&mut self, indices: &[usize], value: f64) {
        let flat_index = self.calculate_flat_index(indices);
        self.data[flat_index] = value;
    }

    fn calculate_flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "ランクが一致しません");
        let mut flat_index = 0;
        let mut stride = 1;
        for i in (0..self.shape.len()).rev() {
            let dim_index = indices[i];
            let dim_size = self.shape[i];
            assert!(dim_index < dim_size, "インデックスが範囲外です: dim {}", i);
            flat_index += dim_index * stride;
            stride *= dim_size;
        }
        flat_index
    }

    fn calculate_multi_index(&self, flat_index: usize) -> Vec<usize> {
        let mut indices = vec![0; self.shape.len()];
        let mut remaining = flat_index;
        for i in (0..self.shape.len()).rev() {
            let dim_size = self.shape[i];
            indices[i] = remaining % dim_size;
            remaining /= dim_size;
        }
        indices
    }

    /// 汎用Einsum実装
    pub fn einsum(formula: &str, inputs: &[&Tensor]) -> Tensor {
        // tensor_idx : どのTensorかを示すインデックス
        // indices    : Tensorの成分のインデックス(usize)の組
        // sss_list   : Tensorの成分の添え字の組のリスト e.g. ["ij","jk"]
        // sss        : Tensorの成分の添え字の組 e.g. "ij"
        // shape      : Tensorの次元の組
        // size       : Tensorのある次元のサイズ
        // dim_idx    : Tensorのある次元番号

        // 1. 式のパースと次元サイズの決定
        let parts: Vec<&str> = formula.split("->").collect();
        let input_fmt = parts[0];
        let output_fmt = parts[1];
        let input_sss_list: Vec<&str> = input_fmt.split(',').collect();
        assert_eq!(input_sss_list.len(), inputs.len(), "入力テンソルの数が一致しません");

        // 2. 出力テンソルの形状決定
        let mut ss2size: HashMap<char, usize> = HashMap::new();
        for (i, raw_sss) in input_sss_list.iter().enumerate() {
            let tensor = inputs[i];
            let sss: Vec<char> = raw_sss.chars().collect();
            assert_eq!(sss.len(), tensor.shape.len(), "入力 {} のランクが一致しません", i);
            for (dim, &ss) in sss.iter().enumerate() {
                let size = tensor.shape[dim];
                if let Some(&prev) = ss2size.get(&ss) {
                    assert_eq!(prev, size, "インデックス {} の次元サイズが不一致です", ss);
                } else {
                    ss2size.insert(ss, size);
                }
            }
        }
        
        let mut output_shape = Vec::new();
        let output_sss: Vec<char> = output_fmt.chars().collect();
        for ss in &output_sss {
            output_shape.push(*ss2size.get(ss).expect("出力インデックスが入力に見つかりません"));
        }
        let mut result = Tensor::zeros(output_shape);

        // 3. ループ実行 (位取り記数法によるカウンタ)
        let loop_sss: Vec<char> = ss2size.keys().cloned().collect();
        let limits: Vec<usize> = loop_sss.iter().map(|ss| ss2size[ss]).collect();
        let mut counters: Vec<usize> = vec![0; loop_sss.len()];
        let ss2idx: HashMap<char, usize> = loop_sss.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        loop {
            // 積の計算
            let mut prod = 1.0;
            for (i, tensor) in inputs.iter().enumerate() {
                let mut indices = Vec::with_capacity(tensor.shape.len());
                for ss in input_sss_list[i].chars() {
                    indices.push(counters[ss2idx[&ss]]);
                }
                prod *= tensor.get(&indices);
            }
            
            // 結果への加算
            let mut out_indices = Vec::with_capacity(result.shape.len());
            for ss in output_sss.iter() {
                out_indices.push(counters[ss2idx[&ss]]);
            }
            let val = result.get(&out_indices);
            result.set(&out_indices, val + prod);

            // カウンタのインクリメント
            let mut carry = true;
            for i in (0..counters.len()).rev() {
                counters[i] += 1;
                if counters[i] < limits[i] {
                    carry = false;
                    break;
                }
                counters[i] = 0;
            }
            if carry { break; }
        }
        result
    }

    /// テンソルを行列スライスの羅列として表示
    #[allow(dead_code)]
    pub fn print_tensor(&self) {
        let rank = self.shape.len();
        if rank < 2 {
            println!("Tensor {:?}: {:?}", self.shape, self.data);
            return;
        }

        let outer_dims = &self.shape[..rank - 2];
        let (rows, cols) = (self.shape[rank - 2], self.shape[rank - 1]);
        let mut counters = vec![0; outer_dims.len()];

        loop {
            if !outer_dims.is_empty() { println!("Slice {:?}:", counters); }
            else { println!("Matrix:"); }

            for r in 0..rows {
                print!("[ ");
                for c in 0..cols {
                    let mut indices = counters.clone();
                    indices.push(r);
                    indices.push(c);
                    let val = self.get(&indices);
                    if val.abs() < 1e-9 { print!("  .  "); } 
                    else { print!("{:^5.2}", val); }
                }
                println!(" ]");
            }
            println!();

            if outer_dims.is_empty() { break; }

            let mut carry = true;
            for i in (0..counters.len()).rev() {
                counters[i] += 1;
                if counters[i] < outer_dims[i] {
                    carry = false;
                    break;
                }
                counters[i] = 0;
            }
            if carry { break; }
        }
    }

    /// テンソルの内容をリレーションテーブル形式でCSV出力する
    /// 
    /// # Arguments
    /// * `path`: 出力先ファイルパス
    /// * `header`: CSVヘッダー行 (例: `&["Subject", "Object"]`)
    /// * `dim_labels`: 各次元のインデックスに対応するラベルのリスト
    pub fn export_relation_csv(&self, path: &str, header: &[&str], dim_labels: &[&[&str]]) {
        assert_eq!(self.shape.len(), dim_labels.len(), "次元数とラベルリストの長さが不一致です");
        assert_eq!(self.shape.len() + 1, header.len(), "ヘッダー列数が (次元数 + 1:値) と一致しません");

        // 親ディレクトリを作成するように修正
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).unwrap_or(());
        }

        let mut file = std::fs::File::create(path).expect("ファイル作成に失敗しました");
        writeln!(file, "{}", header.join(",")).expect("ヘッダー書き込みに失敗しました");

        for (i, &val) in self.data.iter().enumerate() {
            if val.abs() > 1e-9 {
                let indices = self.calculate_multi_index(i);
                let mut row = Vec::new();
                
                // インデックスをラベルに変換
                for (dim, &idx) in indices.iter().enumerate() {
                    let label = if idx < dim_labels[dim].len() {
                        dim_labels[dim][idx]
                    } else {
                        "Unknown"
                    };
                    row.push(label.to_string());
                }
                
                // 値を追加 (1.0なら整数っぽく、それ以外は小数)
                if (val - val.round()).abs() < 1e-9 {
                    row.push(format!("{:.0}", val));
                } else {
                    row.push(format!("{:.2}", val));
                }

                writeln!(file, "{}", row.join(",")).expect("行データの書き込みに失敗しました");
            }
        }
        println!("Exported: {}", path);
    }

    /// テンソルの内容を読みやすい形式で表示するユーティリティ
    pub fn print_nonzero(&self) {
        println!("非ゼロ要素 (Shape: {:?}):", self.shape);
        for (i, &val) in self.data.iter().enumerate() {
            if val.abs() > 1e-9 {
                let indices = self.calculate_multi_index(i);
                println!("  {:?} -> {:.2}", indices, val);
            }
        }
    }
}
