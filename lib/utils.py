import os, io
import numpy as np
import IPython
from PIL import Image, ImageDraw

### 
class DBUtils:
    @staticmethod
    def is_table_exists(db_con, table_name):
        cur = db_con.cursor() 
        return len(cur.execute('SELECT name FROM sqlite_master WHERE type=:type AND name=:table_name', {'type': 'table', 'table_name': table_name}).fetchall()) > 0

    @staticmethod
    def is_table_empty(db_con, table_name):
        cur = db_con.cursor() 
        return len(cur.execute(f'SELECT * FROM {table_name} LIMIT 1').fetchall()) < 1

    @staticmethod
    def drop_table_safe(db_con, tn):
        if DBUtils.is_table_exists(db_con, tn):
            db_con.cursor().execute(f'DROP TABLE {tn}')
            db_con.commit()

    @staticmethod
    def get_full_db_file_name(config, db_file_name, with_prefix=True):
        base_path = os.path.dirname(os.path.abspath(config.config_fname))
        return os.path.join(os.path.join(base_path, config.dataset_path), ('', config.db_file_name_prefix)[with_prefix] + db_file_name)

    @staticmethod
    def get_column_names(db_con, table_name):
        cur = db_con.cursor() 
        return list(map(lambda row: row[1], cur.execute(f'PRAGMA table_info({table_name})').fetchall()))

    @staticmethod
    def ensure_table_columns(db_con, table_name, column_names):
        cur = db_con.cursor() 
        existing_column_names = set(map(lambda row: row[1], cur.execute(f'PRAGMA table_info({table_name})').fetchall()))
        missing_column_names = set(column_names) - existing_column_names

        if not missing_column_names:
            return

        for column_name in missing_column_names:
            cur.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name}')

        db_con.commit()

###
class MathUtils:
    @staticmethod
    def softmax(x):
        max_x = np.max(x)
        exp_x = np.exp(x - max_x)
        sum_exp_x = np.sum(exp_x)
        return exp_x / sum_exp_x

    @staticmethod
    def conflate(pdfs):    
        n = np.prod(pdfs, axis=0)
        d = n.sum()
    
        if np.isclose(d, 0):
            return np.zeros(len(pdfs))
            
        return n / d

    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret[:n-1] = np.array(a[:n-1]) * n
        return ret / n

    @staticmethod
    def get_angle_diff(a_from, a_to):
        andiff = a_to - a_from
        return (andiff + 180) % 360 - 180

### 
# from https://gist.github.com/parente/691d150c934b89ce744b5d54103d7f1e
def _html_src_from_raw_image_data(data):
    """Base64 encodes image bytes for inclusion in an HTML img element"""
    img_obj = IPython.display.Image(data=data)
    for bundle in img_obj._repr_mimebundle_():
        for mimetype, b64value in bundle.items():
            if mimetype.startswith('image/'):
                return f'data:{mimetype};base64,{b64value}'

def display_images(images, captions=None, row_height='auto'):
    figures = []
    
    for image_index, image in enumerate(images):
        if isinstance(image, bytes) or isinstance(image, Image.Image):
            if isinstance(image, bytes):
                bts = image
            else:
                b = io.BytesIO()
                image.save(b, format='PNG')
                bts = b.getvalue()
            
            src = _html_src_from_raw_image_data(bts)
        else:
            src = image
            #caption = f'<figcaption style="font-size: 0.6em">{image}</figcaption>'

        caption = ''
        
        if not captions is None:
            if isinstance(captions, dict):
                caption = captions.get(id(image), '')
            else:
                assert len(captions) == len(images)
                caption = captions[image_index]

            if caption:
                caption = f'<figcaption style="font-size: 0.6em">{caption}</figcaption>'
        
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{src}" style="height: {row_height}">
              {caption}
            </figure>
        ''')
    return IPython.display.HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')

def display_images_grid(images, col_count, col_width=None, captions=None):
    figures = []
    
    for image_index, image in enumerate(images):
        assert isinstance(image, bytes) or isinstance(image, Image.Image)

        if isinstance(image, bytes):
            bts = image
        else:
            b = io.BytesIO()
            image.save(b, format='PNG')
            bts = b.getvalue()
        
        src = _html_src_from_raw_image_data(bts)

        caption = ''

        if not captions is None:
            if isinstance(captions, dict):
                caption = str(captions.get(id(image), ''))
            else:
                assert len(captions) == len(images), (len(captions), len(images))
                caption = str(captions[image_index])

            if caption:
                caption = f'<figcaption style="font-size: 0.6em">{caption}</figcaption>'
        
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{src}" style="height: auto">
              {caption}
            </figure>
        ''')

    if not col_width:
        if len(images) > 0 and isinstance(images[0], Image.Image):
            col_width = images[0].width

    if not col_width: 
        col_width='auto'
    else:
        col_width = f'{col_width}px'
        
    return IPython.display.HTML(data=f'''<div style="
        display: grid; 
        grid-template-columns: repeat({col_count}, {col_width});
        column-gap: 1px;
        row-gap: 1px;">
        {''.join(figures)}
    </div>''')

def vec_to_square_matrix(v):
    sz = int(np.sqrt(v.shape[0]))
    assert sz * sz == v.shape[0]
    return v.reshape(sz, -1)

def matrix_to_image(m):
    m = m.ravel()
    sz = int(np.sqrt(m.shape[0]))
    assert sz * sz == m.shape[0]
    return Image.frombytes('L', size=(sz, sz), data=m.astype('b'))

def lay_grid(image, step=16):
    draw = ImageDraw.Draw(image)

    for c in range(step - 1, image.height, step):
        draw.line([0, c, image.width, c], fill='gray')
        draw.line([c, 0, c, image.height], fill='gray')

    return image