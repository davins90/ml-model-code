import kfp
from kfp.v2 import dsl
from kfp.components import load_component_from_file
from kfp.v2 import compiler

PROJECT_ID = 'test-mlops-pipeline-438622'  # Sostituisci con il tuo project ID
BUCKET_NAME = 'ml-model-bucket-mlops'  # Sostituisci con il tuo bucket

train_component = load_component_from_file('train_component.yaml')

@dsl.pipeline(
    name='Iris Classification Pipeline',
    description='Pipeline to train an Iris classification model.'
)
def iris_pipeline(bucket_name: str = BUCKET_NAME):
    train_task = train_component(bucket_name)

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path='iris_pipeline.json'
    )
