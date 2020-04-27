"""
Utilities to steamline common sagemaker setup tasks.
"""

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

