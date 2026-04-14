import os
import io
import string
import requests
import time

from utils import *
from logging_utils import *

class ModelRegistry:
    def __init__(self, maven_group_id, nexus_url='http://nexus:8081', download_nexus_url=None, nexus_auth=('bot', 'bot'), maven_repo='model-registry'):
        self.maven_group_id = maven_group_id
        self.nexus_url = nexus_url
        self.download_nexus_url = LangUtils.coalesce(download_nexus_url, self.nexus_url)
        self.nexus_auth = nexus_auth
        self.maven_repo = maven_repo

    def list_models(self):
        query_params = {
            'group': self.maven_group_id, 
        }
        r = requests.get(f'{self.nexus_url}/service/rest/v1/search', params=query_params, auth=self.nexus_auth)
        r.raise_for_status()
        r_json = r.json()
        models = []
        
        while True:
            items = r.json()['items']
            models.extend(map(lambda item: dict(name=item['name'], version=item['version']), items))
            continuation_token = r_json.get('continuationToken', '')
        
            if not continuation_token:
                break
                
            query_params['continuationToken'] = continuation_token
            r = requests.get(f'{self.nexus_url}/service/rest/v1/search', params=query_params, auth=self.nexus_auth)
            r.raise_for_status()
            r_json = r.json()

        return models
            
    def register_model(self, model_name, model_version):
        query_params = {
            'repository': self.maven_repo,
        }
        form_data = {
            'maven2.generate-pom': False,
            'maven2.groupId': self.maven_group_id,
            'maven2.artifactId': model_name,
            'version': model_version,
        }
        pom = '''
<project 
xmlns="http://maven.apache.org/POM/4.0.0" 
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
<modelVersion>4.0.0</modelVersion>
<groupId>${MODEL_GROUP_URI}</groupId>
<artifactId>${MODEL_NAME}</artifactId>
<version>${MODEL_VERSION}</version>
</project>'''
        expandvars = dict(
            MODEL_GROUP_URI=self.maven_group_id,
            MODEL_NAME=model_name,
            MODEL_VERSION=model_version,
        )
        pom = string.Template(pom).safe_substitute(expandvars)

        with io.StringIO(pom) as main_model_asset:
            main_model_asset.seek(0)
            files = {'maven2.asset1': main_model_asset}
            form_data['maven2.asset1.extension'] = 'pom'
            r = requests.post(f'{self.nexus_url}/service/rest/v1/components', auth=self.nexus_auth, params=query_params, data=form_data, files=files)
            r.raise_for_status()

    def attach_asset(self, model_name, model_version, asset, asset_ext='', asset_classifier='', replace=False):
        query_params = {
            'repository': self.maven_repo,
        }
        form_data = {
            'maven2.generate-pom': False,
            'maven2.groupId': self.maven_group_id,
            'maven2.artifactId': model_name,
            'version': model_version,
        }

        if asset_classifier:
            form_data['maven2.asset1.classifier'] = asset_classifier
        
        files = {}
        do_post = lambda: requests.post(f'{self.nexus_url}/service/rest/v1/components', auth=self.nexus_auth, params=query_params, data=form_data, files=files)
        
        if isinstance(asset, str):
            with open(asset, 'rb') as asset_file:
                files['maven2.asset1'] = asset_file
                form_data['maven2.asset1.extension'] = os.path.splitext(asset)[1].lstrip('.')
                r = do_post()
        elif isinstance(asset, io.IOBase):
            assert asset_ext, 'asset_ext arg must be specified'
            asset.seek(0)
            files['maven2.asset1'] = asset
            form_data['maven2.asset1.extension'] = asset_ext
            r = do_post()
        else:
            assert False, f'Unsupported asset type={type(asset)}'

        if r.status_code == 400 and replace:
            assets = self.filter_assets(self.get_assets(model_name, model_version), form_data['maven2.asset1.extension'], asset_classifier)

            if assets:
                Logging.info(f'Found existing {self.describe_asset(form_data['maven2.asset1.extension'], asset_classifier)} asset (id={assets[0]['id']}) ' + 
                             f'for {self.maven_group_id}.{model_name}:{model_version}, replacing')
                r = requests.delete(f'{self.nexus_url}/service/rest/v1/assets/{assets[0]['id']}', auth=self.nexus_auth)
                r.raise_for_status()
                self.attach_asset(model_name, model_version, asset, asset_ext, asset_classifier, replace=False)
            else:
                r.raise_for_status()
                assert False
        else:
            r.raise_for_status()
            Logging.info(f'{self.describe_asset(form_data['maven2.asset1.extension'], asset_classifier)} asset attached to {self.maven_group_id}.{model_name}:{model_version}')

    def get_assets(self, model_name, model_version):
        query_params = {
            'group': self.maven_group_id, 
            'name': model_name,
            'version': model_version,
        }
        r = requests.get(f'{self.nexus_url}/service/rest/v1/search', params=query_params, auth=self.nexus_auth)
        r.raise_for_status()
        items = r.json()['items']

        if not items:
            return []

        return items[0]['assets']

    def get_asset_content(self, model_name, model_version, asset_ext, asset_classifier=''):
        assert asset_ext, 'asset_ext arg must be specified'
        assets = self.get_assets(model_name, model_version)
        assets = self.filter_assets(assets, asset_ext, asset_classifier)

        if not assets:
            raise Exception(f'Failed to locate asset {self.describe_asset(asset_ext, asset_classifier)} for {self.maven_group_id}.{model_name}:{model_version}')

        # By default artifact is available for download via `assets[0]['downloadUrl']`. But we would consturct
        # download URL manually in order to benefit from caching nexus repo (if present, OFC)
        download_url  = self.download_nexus_url + LangUtils.when(self.download_nexus_url.endswith('/'), '', '/') 
        download_url += f'repository/{self.maven_repo}/'
        download_url += assets[0]['path']
        Logging.debug(f'Downloading {download_url} for asset with id={assets[0]['id']}')
        r = requests.get(download_url)
        r.raise_for_status()
        return r.content

    def is_asset_present(self, model_name, model_version, asset_ext, asset_classifier=''):
        assert asset_ext, 'asset_ext arg must be specified'
        assets = self.get_assets(model_name, model_version)
        assets = self.filter_assets(assets, asset_ext, asset_classifier)
        return len(assets) > 0

    def describe_asset(self, asset_ext, asset_classifier):
        result = ''
        
        if asset_classifier:
            result += '.' + asset_classifier

        return result + '.' + asset_ext

    def filter_assets(self, assets, asset_ext, asset_classifier):
        is_ext_match = lambda a: a['maven2']['extension'] == asset_ext # mandatory match
        filter_func = is_ext_match

        if asset_classifier:
            is_classifier_match = lambda a: a['maven2'].get('classifier', '') == asset_classifier # optional match
            filter_func = lambda a: is_ext_match(a) and is_classifier_match(a)
            
        return list(filter(filter_func, assets))     

