import os
import configparser

class Config:
    DEFAULTS = dict(
        hdc_n=10000,
        sample_size=128,
        cortical_columns_count=1,
        cortical_column_receptive_field_size=128, # for 'random' this is the number of sensor groups, for 'radial' - radius
        cortical_columns_layout='random',
        encoding_type='normal',
        sensor_groups_count=256,
        sensors_count=256,
        retina_layout='grid',
        dataset_source='dataset_source',
        dataset_path='dataset',
        dataset_sample_count=10000,
        dataset_train_samples_count=10000,
        dataset_test_samples_count=2000,
        dataset_metadata_file_name='_metadata.json',
        output_path='out',
        db_file_name_prefix='',
        hdv_db_file_name='hdv.db',
        train_db_file_name='train.db',
        test_db_file_name='test.db',
        test_results_db_file_name='test_results.db',
        transfs_db_file_name='transfs.db',
    )
    
    def __init__(self, section_name='DEFAULT', config_fname='config.txt'):
        super()
        self.section_name = section_name
        self.config_fname = config_fname
        self.reload()

    def reload(self):
        config = configparser.ConfigParser(defaults=type(self).DEFAULTS)

        if os.path.exists(self.config_fname):
            config.read(self.config_fname)

        sections = [config['DEFAULT']]

        if self.section_name != 'DEFAULT':
            sections.append(config[self.section_name])

        for section in sections:
            for k in section:
                typ = type(type(self).DEFAULTS[k]) # enforce proper type (e.g. kernel_size must by int, not string)
                setattr(self, k, typ(section[k]))
