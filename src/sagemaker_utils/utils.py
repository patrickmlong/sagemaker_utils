"""
Utilities to steamline common sagemaker setup tasks.
"""

import numpy as np
import boto3
import sagemaker


def get_credentials():
    """Retrieve and return basic sagemaker credentils: session, role, S3 bucket"""
    sage_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = sage_session.default_bucket()
    return sage_session, role, bucket


def fit_deploy(model, data, instance_count: int = 1, instance_type = "ml.t2.medium"):
    """Fit and deploy a Sagemaker model with default settings
    Args:
         model: Instantiated AWS model
         data: AWS record_set
         instance_count (int): number of Sagemaker instances for model deployment
         instance_type: Type of AWS Sagemaker instance.
     Return:
         Deployed model instance
    """
    model.fit(data)
    deployed_model = model.deploy(initial_instance_count = instance_count,instance_type = instance_type)
    return deployed_model


def s3_path(bucket, prefix):
    """Returns a formatted S3 bucket output path"""
    return f"s3://{bucket}/{prefix}"


def create_record_set(model, X, y, supervised: bool = True):
    """Create a record set for AWS model training"""
    X_float = X.astype("float32")
    if supervised:
        y_float = y.astype("float32")
        return  model.record_set(X_float, labels = y_float)
    else:
        return model.record_set(X_float)
