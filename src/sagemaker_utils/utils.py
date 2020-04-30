"""
Utilities to steamline common AWS sagemaker setup tasks.
"""

import numpy as np
import logging
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


def create_record_set(model, X, y):
    """Create a record set for AWS model training"""
    X_float = X.astype("float32")
    if y:
        y_float = y.astype("float32")
        return  model.record_set(X_float, labels = y_float)
    else:
        return model.record_set(X_float)


def remove_endpoint(endpoint_name):
    """Remove AWS Sagemaker endpoint"""
    try:
        boto3.client("sagemaker").delete_endpoint(EndpointName = endpoint_name)
        logging.info(f"Endpoint: {endpoint_name} deleted")
    except:
        logging.info("Endpoint: {endpoint_name} previously deleted")


def my_aws_region():
    """"Return your AWS region"""
    return boto3.session.Session().region_name


def make_s3_bucket(s3_resource, name):
    """"Create an S3 bucket with a user defined bucket name
    Args:
         s3_resource: AWS s3 client connection e.g. boto3.resource("s3")
         name: unique name for your aws bucket
    Return:
        AWS s3 bucket
    """
    s3_bucket = s3_resource.createBucket(name,
    CreateBucketConfiguration = {"LocationConstraint": my_aws_region()})
    logging.info("S3 bucket created")
    return s3_bucket
