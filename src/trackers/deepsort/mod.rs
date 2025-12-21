use crate::traits::AppearanceExtractor;

pub struct DeepSort<E: AppearanceExtractor> {
    #[allow(dead_code)]
    extractor: E,
}

impl<E: AppearanceExtractor> DeepSort<E> {
    pub fn new(extractor: E) -> Self {
        Self { extractor }
    }
}
