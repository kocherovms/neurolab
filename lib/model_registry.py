import os
import io
import string
import requests
import time

class ModelRegistry:
    def __init__(self, maven_group_id, nexus_url='http://nexus:8081', nexus_auth=('bot', 'bot'), maven_repo='model-registry'):
        self.maven_group_id = maven_group_id
        self.nexus_url = nexus_url
        self.nexus_auth = nexus_auth
        self.maven_repo = maven_repo
            
    def register_model(self, model_name, retries_count=3):
        for retry_no in range(retries_count):
            query_params = {
                'group': self.maven_group_id, 
                'name': model_name,
                'sort': 'version',
                'direction': 'desc'
            }
            r = requests.get(f'{self.nexus_url}/service/rest/v1/search', params=query_params, auth=self.nexus_auth)
            r.raise_for_status()
            items = r.json()['items']
            
            if not items:
                version = 1
                print(f'No versions of {self.maven_group_id}.{model_name} found, starting new series with version={version}')
            else:
                top_version = int(items[0]['version'])
                version = top_version + 1
                print(f'Found {self.maven_group_id}.{model_name}:{top_version}, continuing series with version={version}')
            
            query_params = {
                'repository': self.maven_repo,
            }
            form_data = {
                'maven2.generate-pom': False,
                'maven2.groupId': self.maven_group_id,
                'maven2.artifactId': model_name,
                'version': version,
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
                MODEL_VERSION=version,
            )
            pom = string.Template(pom).safe_substitute(expandvars)
    
            with io.StringIO(pom) as main_model_asset:
                main_model_asset.seek(0)
                files = {'maven2.asset1': main_model_asset}
                form_data['maven2.asset1.extension'] = 'pom'
                r = requests.post(f'{self.nexus_url}/service/rest/v1/components', auth=self.nexus_auth, params=query_params, data=form_data, files=files)

                if r.status_code == 400:
                    # possible race condition
                    print(f'Got Bad Requets status (race condition?), retrying in {retry_no+1} seconds')
                    time.sleep(retry_no + 1)
                    continue
                
            return version

    def attach_asset(self, model_name, model_version, model_asset, model_asset_ext='', replace=False):
        query_params = {
            'repository': self.maven_repo,
        }
        form_data = {
            'maven2.generate-pom': False,
            'maven2.groupId': self.maven_group_id,
            'maven2.artifactId': model_name,
            'version': model_version,
        }
        files = {}
        do_post = lambda: requests.post(f'{self.nexus_url}/service/rest/v1/components', auth=self.nexus_auth, params=query_params, data=form_data, files=files)
        
        if isinstance(model_asset, str):
            with open(model_asset, 'rb') as asset_file:
                files['maven2.asset1'] = asset_file
                form_data['maven2.asset1.extension'] = os.path.splitext(model_asset)[1].lstrip('.')
                r = do_post()
        elif isinstance(model_asset, io.IOBase):
            assert model_asset_ext, 'model_asset_ext arg must be specified'
            model_asset.seek(0)
            files['maven2.asset1'] = model_asset
            form_data['maven2.asset1.extension'] = model_asset_ext
            r = do_post()
        else:
            assert False, f'Unsupported asset type={type(asset)}'

        if r.status_code == 400 and replace:
            assets = list(filter(lambda a: a['maven2']['extension'] == form_data['maven2.asset1.extension'], self.get_assets(model_name, model_version)))

            if assets:
                print(f'Found existing .{form_data['maven2.asset1.extension']} asset (id={assets[0]['id']}) for {self.maven_group_id}.{model_name},version={model_version}, replacing')
                r = requests.delete(f'{self.nexus_url}/service/rest/v1/assets/{assets[0]['id']}', auth=self.nexus_auth)
                r.raise_for_status()
                self.attach_asset(model_name, model_version, model_asset, model_asset_ext, replace=False)
            else:
                r.raise_for_status()
                assert False
        else:
            r.raise_for_status()
            print(f'.{form_data['maven2.asset1.extension']} asset attached to {self.maven_group_id}.{model_name}:{model_version}')

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

    def get_asset_content(self, model_name, model_version, asset_ext):
        assets = self.get_assets(model_name, model_version)
        assets = list(filter(lambda a: a['maven2']['extension'] == asset_ext, assets))

        if not assets:
            raise Exception(f'Failed to locate asset "{asset_ext}" for {self.maven_group_id}.{model_name},version={model_version}')

        print(f'Downloading {assets[0]['downloadUrl']} for asset with id={assets[0]['id']}')
        r = requests.get(assets[0]['downloadUrl'])
        r.raise_for_status()
        return r.content       
        