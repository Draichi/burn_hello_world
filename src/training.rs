use crate::model::Model;
use burn::train::ClassificationOutput;
use burn::{
    tensor::{
        backend::Backend,
        Tensor, Int
    },
    nn::loss::CrossEntropyLoss
};

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}