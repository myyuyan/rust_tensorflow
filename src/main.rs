use ndarray::array;
use ndarray::Array2;
use neural_network::NeuralNetwork;

fn main() {
    let mut nn = NeuralNetwork::new(2, 2, 1, 0.5);

    // 训练数据（逻辑异或问题）
    let inputs = array![[0.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [1.0, 1.0]].reversed_axes();

    let targets = array![[0.0],
                         [1.0],
                         [1.0],
                         [0.0]].reversed_axes();

    // 训练网络
    for _ in 0..10000 {
        nn.train(&inputs, &targets);
    }

    // 测试网络
    let test_input = array![[0.0], [1.0]];
    let output = nn.predict(&test_input);
    println!("Output: {:?}", output);
}
