import os
import requests
import io

class ModelRegistry:
    def __init__(self, maven_group_id, nexus_url='http://nexus:8081', nexus_auth=('bot', 'bot'), maven_repo='model-registry'):
        self.maven_group_id = maven_group_id
        self.nexus_url = nexus_url
        self.nexus_auth = nexus_auth
        self.maven_repo = maven_repo
            
    def register_model(self, model_name, main_model_asset, main_model_asset_ext=''):
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
            print(f'Top version of {self.maven_group_id}.{model_name} is {top_version}, continuing series with version={version}')
        
        query_params = {
            'repository': self.maven_repo,
        }
        form_data = {
            'maven2.generate-pom': True,
            'maven2.groupId': self.maven_group_id,
            'maven2.artifactId': model_name,
            'version': version,
        }
        files = {}
        do_post = lambda: requests.post(f'{self.nexus_url}/service/rest/v1/components', auth=self.nexus_auth, params=query_params, data=form_data, files=files)

        if isinstance(main_model_asset, str):
            with open(main_model_asset, 'r') as asset_file:
                files['maven2.asset1'] = asset_file
                form_data['maven2.asset1.extension'] = os.path.splitext(main_model_asset)[1].lstrip('.')
                r = do_post()
        elif isinstance(main_model_asset, io.IOBase):
            assert main_model_asset_ext, 'main_model_asset_ext arg must be specified'
            main_model_asset.seek(0)
            files['maven2.asset1'] = main_model_asset
            form_data['maven2.asset1.extension'] = main_model_asset_ext
            r = do_post()
        else:
            assert False, f'Unsupported asset type={type(asset)}'
        
        r.raise_for_status()
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
                print(f'Found existing .{form_data['maven2.asset1.extension']} asset (id={assets[0]['id']}) for {self.maven_group_id}.{model_name}.{model_version}, replacing')
                r = requests.delete(f'{self.nexus_url}/service/rest/v1/assets/{assets[0]['id']}', auth=self.nexus_auth)
                r.raise_for_status()
                self.attach_asset(model_name, model_version, model_asset, model_asset_ext, replace=False)
            else:
                r.raise_for_status()
                assert False
        else:
            r.raise_for_status()
            print(f'.{form_data['maven2.asset1.extension']} asset attached to {self.maven_group_id}.{model_name}.{model_version}')

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