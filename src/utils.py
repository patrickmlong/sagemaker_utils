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
