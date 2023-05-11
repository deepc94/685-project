import datasets
import pathlib
import json

# Dataset Info

_HOMEPAGE = 'homepage for the dataset (e.g. github repo)'

_VERSION = '1.0.0'

_LICENSE = '(MIT?)'

_CITATION = '''Bibtex citation
'''

_DESCRIPTION = '''Demo dataset for demo purposes.
'''

_REPO = 'huggingface dataset repo'

_ATTRIBUTES = [
    'shape',
    'size',
    'color'
]

_ATTR_VALUES = [
    'same', 'different'
]

_BASE_URL = 'data.zip'

_IMG_DIR = 'data'

_METADATA_URLS = {
    'train': 'metadata/train.json',
    'test': 'metadata/test.json'
}

class Dataset(datasets.GeneratorBasedBuilder):
    '''Dataset for NLP Course final project'''

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'image' : datasets.Image(),
                    'shape' : datasets.ClassLabel(names=_ATTR_VALUES),
                    'size': datasets.ClassLabel(names=_ATTR_VALUES),
                    'color' : datasets.ClassLabel(names=_ATTR_VALUES)
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            version=_VERSION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        archive_path = dl_manager.download(_BASE_URL)
        split_metadata_paths = dl_manager.download(_METADATA_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'images': dl_manager.iter_archive(archive_path),
                    'metadata_path': split_metadata_paths['train'],
                    'split': 'train'
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'images': dl_manager.iter_archive(archive_path),
                    'metadata_path': split_metadata_paths['test'],
                    'split': 'test'
                },
            ),
        ]

    def _generate_examples(self, images, metadata_path, split):
        '''Generate images and labels for splits.'''
        with open(metadata_path, encoding='utf-8') as f:
            metadata = json.load(f)
        for file_path, file_obj in images:
            file_path_parts = pathlib.Path(file_path).parts
            if (file_path_parts[0]==_IMG_DIR) and (file_path_parts[1]==split):
                filename = file_path_parts[2]
                if filename in metadata:
                    yield file_path, {
                        'image': {'path': file_path, 'bytes': file_obj.read()},
                        'shape': metadata[filename]['shape'],
                        'size': metadata[filename]['size'],
                        'color': metadata[filename]['color']
                    }
