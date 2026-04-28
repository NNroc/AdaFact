import os
import argparse
import logging
import io
import fitz
from natsort import natsorted
from PIL import Image
from pdf2image import convert_from_path

from config.base_config import global_config
from src.model.mrag import Model
from src.shared_parser import get_shared_parser
from utils.chunking_utils import split_content
from utils.decorator_utils import parallel_processor
from utils.file_utils import load_json, prepare_files
from utils.pdf2markdown_utils import split_md_by_page, pdf2markdown_mineru


def convert_pdf_to_images(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        img_data = pix.tobytes("png")
        images.append(Image.open(io.BytesIO(img_data)))
    doc.close()
    return images


def encode_document_all(index_model, doc_path, doc_id, doc_md_path, mineru_only=False, debug=False):
    # mineru 切分
    folder_path = pdf2markdown_mineru(doc_path, doc_md_path, doc_id)
    if mineru_only:
        return
    files = os.listdir(folder_path)
    markdown_files = [file for file in files if file.endswith(".md")]
    page_images = convert_from_path(doc_path, dpi=144)
    page_images_information = []
    os.makedirs(f'{index_model.working_dir}/page_image/', exist_ok=True)
    for page_num, page_snapshot in enumerate(page_images):
        image_path = f"{index_model.working_dir}/page_image/{doc_id}-{page_num + 1}.png"
        page_images_information.append({'image_path': f"/page_image/{doc_id}-{page_num + 1}.png", 'page_idx': page_num})
        # re-index the page number to start from 1
        if not os.path.exists(image_path):
            page_snapshot.save(image_path)

    if len(markdown_files) != 1:
        raise ValueError(
            f'No unique .md file was found in the folder {doc_id}. Please ensure there is only one .md file in the folder.')
    markdown_file_path = f'{folder_path}/{markdown_files[0]}'
    content_list_file_path = f'{folder_path}/{doc_id}_content_list.json'
    content_list = load_json(content_list_file_path)

    content_image_list = split_content(content_list)
    full_text_with_page = split_md_by_page(markdown_file_path, content_list_file_path)
    full_text, pages_idx = [], []
    for page_idx in full_text_with_page:
        full_text.append(full_text_with_page[page_idx])
        pages_idx.append(page_idx + 1)

    page_information = {p_id + 1: {'token_num': 0, 'image_num': 0} for p_id in range(len(page_images))}
    for content_img in content_image_list:
        if len(content_img['img_path']) > 5:
            page_information[content_img['page_idx'] + 1]['image_num'] += 1

    index_model.insert_origin_page_image(page_images_information)
    # index_model.insert_text(full_text, pages_idx)
    # information_text = index_model.chunks_information_text.get_data()
    # for chunk_id, chunk in information_text.items():
    #     page_information[chunk['page_idx']]['token_num'] += chunk['tokens']
    # index_model.insert_image(content_image_list, f"{folder_path}/", page_information)

    # index_model.information_fusion()
    index_model.generate_embeddings(page_images)


def encode_document(index_model, doc_path, doc_id, debug=False):
    page_images = convert_from_path(doc_path, dpi=144)
    page_images_information = []
    os.makedirs(f'{index_model.working_dir}/page_image/', exist_ok=True)
    for page_num, page_snapshot in enumerate(page_images):
        image_path = f"{index_model.working_dir}/page_image/{doc_id}-{page_num + 1}.png"
        page_images_information.append({'image_path': f"/page_image/{doc_id}-{page_num + 1}.png", 'page_idx': page_num})
        # re-index the page number to start from 1
        if not os.path.exists(image_path):
            page_snapshot.save(image_path)

    index_model.insert_origin_page_image(page_images_information)
    index_model.generate_embeddings(page_images)


def encode_images_from_folder(index_model, folder_path, doc_id):
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = natsorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(supported_extensions)
    ])

    page_images = []
    page_images_information = []
    os.makedirs(f'{index_model.working_dir}/page_image/', exist_ok=True)
    for page_num, file_name in enumerate(image_files):
        img_full_path = os.path.join(folder_path, file_name)
        page_snapshot = Image.open(img_full_path).convert("RGB")
        page_images.append(page_snapshot)
        page_images_information.append({'image_path': f"/page_image/{doc_id}-{page_num + 1}.png", 'page_idx': page_num})

    index_model.insert_origin_page_image(page_images_information)
    index_model.generate_embeddings(page_images)


@parallel_processor(mode="thread", max_workers=16, desc='Index')
def index(doc_path):
    doc_id = str(doc_path).replace(".pdf", "")
    working_dir = f"{args.databases_save_dir}/{args.dataset}{args.prefix}/{doc_id}"
    index_model = Model(working_dir=working_dir, dataset=args.dataset)
    encode_document(index_model=index_model,
                    doc_path=f"{args.dataset_dir}/{args.dataset}/{doc_path}/page_image",
                    doc_id=doc_id,
                    debug=args.debug)
    # encode_document_all(index_model=index_model,
    #                 doc_path=f"{args.dataset_dir}/{args.dataset}/{doc_path}",
    #                 doc_id=doc_id,
    #                 doc_md_path=f"{args.minure_save_dir}/{args.dataset}",
    #                 mineru_only=args.mineru_only,
    #                 debug=args.debug)


@parallel_processor(mode="thread", max_workers=1, desc='Index')
def index_folder(folder_path):
    doc_id = folder_path
    working_dir = f"{args.databases_save_dir}/{args.dataset}{args.prefix}/{doc_id}"
    index_model = Model(working_dir=working_dir, dataset=args.dataset)
    encode_images_from_folder(index_model=index_model,
                              folder_path=f"{args.dataset_dir}/{args.dataset}/{folder_path}/page_image",
                              doc_id=doc_id)


if __name__ == "__main__":
    shared_parser = get_shared_parser()
    parser = argparse.ArgumentParser(parents=[shared_parser])
    parser.add_argument('--mineru_only', action='store_true', help="only use mineru, no index")
    args = parser.parse_args()
    if args.debug:
        global_config.logger.setLevel(logging.DEBUG)
    global_config.update_config(args)

    # documents = prepare_files(f"{args.dataset_dir}/{args.dataset}", suffix=".pdf")
    # index(documents)
    dataset_folder_path = f"{args.databases_save_dir}/{args.dataset}"
    dataset_folder_names = [
        f for f in os.listdir(dataset_folder_path)
        if os.path.isdir(os.path.join(dataset_folder_path, f))
    ]
    index_folder(dataset_folder_names)
