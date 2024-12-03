use ndarray::{Array2, Array};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    learning_rate: f64,
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let weights_input_hidden = Array2::random((hidden_size, input_size), Uniform::new(-1.0, 1.0));
        let weights_hidden_output = Array2::random((output_size, hidden_size), Uniform::new(-1.0, 1.0));

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            learning_rate,
            weights_input_hidden,
            weights_hidden_output,
        }
    }

    fn activation(x: &Array2<f64>) -> Array2<f64> {
        1.0 / (1.0 + (-x).mapv(f64::exp))
    }

    fn activation_derivative(x: &Array2<f64>) -> Array2<f64> {
        x * &(1.0 - x)
    }

    pub fn train(&mut self, inputs: &Array2<f64>, targets: &Array2<f64>) {
        // 前向传播
        let hidden_inputs = self.weights_input_hidden.dot(inputs);
        let hidden_outputs = Self::activation(&hidden_inputs);

        let final_inputs = self.weights_hidden_output.dot(&hidden_outputs);
        let final_outputs = Self::activation(&final_inputs);

        // 计算误差
        let output_errors = targets - &final_outputs;
        let hidden_errors = self.weights_hidden_output.t().dot(&output_errors);

        // 反向传播
        self.weights_hidden_output += &self.learning_rate
            * output_errors
            * &Self::activation_derivative(&final_outputs)
            .dot(&hidden_outputs.t());

        self.weights_input_hidden += &self.learning_rate
            * hidden_errors
            * &Self::activation_derivative(&hidden_outputs)
            .dot(&inputs.t());
    }

    pub fn predict(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let hidden_inputs = self.weights_input_hidden.dot(inputs);
        let hidden_outputs = Self::activation(&hidden_inputs);

        let final_inputs = self.weights_hidden_output.dot(&hidden_outputs);
        let final_outputs = Self::activation(&final_inputs);

        final_outputs
    }
}
