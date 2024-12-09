from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
import os
import json
from torch.utils.data import DataLoader

class LAIONSceneGraphDataset(Dataset):
    def __init__(self, image_dir, text_json_path, image_size=(256, 256), max_objects=10, max_samples=None,
                 include_relationships=True, use_orphaned_objects=True):
        super(LAIONSceneGraphDataset, self).__init__()

        self.image_dir = image_dir
        self.image_size = image_size[0]
        self.text_json_path = text_json_path

        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.max_samples = max_samples
        self.include_relationships = include_relationships

        self.text_json_data = self.my_load_json()

        transform = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform)


    def __len__(self):
        num = len(self.text_json_data)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def __getitem__(self, index):

        json_data = self.text_json_data[index]

        img_path = os.path.join(self.image_dir, str(json_data['name']))

        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                width, height = image.size
                image = image.resize((self.image_size, self.image_size), PIL.Image.BILINEAR)
                image = self.transform(image.convert('RGB'))


        triples = []
        global_ids = []
        related_items = set()

        items = json_data["items"]
        relations = json_data["relations"]

        img_id = json_data["img_id"]

        item_dict = {item["item_id"]: item for item in items}

        for relation in relations:
            item1_id = relation["item1"]
            item2_id = relation["item2"]

            related_items.add(item1_id)
            related_items.add(item2_id)

            item1_attributes = " ".join(item_dict[item1_id]["attributes"]) + " " + item_dict[item1_id]["label"]
            item2_attributes = " ".join(item_dict[item2_id]["attributes"]) + " " + item_dict[item2_id]["label"]

            triples.append({
                "item1": item1_attributes,
                "relation": relation["relation"],
                "item2": item2_attributes
            })

            global_ids.append({
                "item1": item_dict[item1_id]["global_item_id"],
                "item2": item_dict[item2_id]["global_item_id"]
            })


        isolated_items = []
        for item in items:
            if item["item_id"] not in related_items:
                isolated_items.append(" ".join(item["attributes"]) + " " + item["label"])

        img_size = (width, height)

        img_text_prompt = json_data['caption_ori']

        return image, triples, global_ids, isolated_items, img_text_prompt, img_size, img_id

    def my_load_json(self):
        with open(self.text_json_path, 'r') as file:
            json_data = json.load(file)

        filtered_json_data = [entry for entry in json_data if entry["relations"]]

        return filtered_json_data

    def my_load_json_old(self):
        with open(self.text_json_path, 'r') as file:
            json_data = json.load(file)

        filtered_json_data = []

        for entry in json_data:
            if entry["relations"]:
                img_path = os.path.join(self.image_dir, str(entry['name']))
                try:
                    with open(img_path, 'rb') as f:
                        with PIL.Image.open(f) as image:
                            image.convert('RGB')
                    filtered_json_data.append(entry)
                except (OSError, PIL.UnidentifiedImageError) as e:
                    print(f"Skipping entry due to image error: {img_path}, error: {e}")
        return filtered_json_data

def my_laion_collate_fn(batch):
    all_imgs, all_triples, all_global_ids, all_isolated_items, = [], [], [], []

    all_text_prompts = []

    all_original_sizes = []

    all_crop_top_lefts = []

    all_img_ids = []

    for i, (image, triples, global_ids, isolated_items, img_text_prompt, img_size,img_id) in enumerate(batch):
        all_imgs.append(image[None])

        all_triples.append(triples)

        all_global_ids.append(global_ids)

        all_original_sizes.append(img_size)

        all_crop_top_lefts.append((0, 0))

        all_isolated_items.append(isolated_items)

        all_text_prompts.append(img_text_prompt)

        all_img_ids.append(img_id)

    out = (all_imgs, all_triples, all_global_ids, all_isolated_items,
           all_text_prompts, all_original_sizes, all_crop_top_lefts,all_img_ids)
    return out

def build_laion_dsets(args):
    dset_kwargs = {
        'image_dir': args.image_dir,
        'text_json_path': args.train_json_path,
        'image_size': (args.image_size, args.image_size),
        'max_samples': None,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.use_orphaned_objects,
        'include_relationships': args.include_relationships,
    }
    train_dset = LAIONSceneGraphDataset(**dset_kwargs)


    iter_per_epoch = len(train_dset) // args.batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    dset_kwargs['text_json_path'] = args.val_json_path

    print(args.val_json_path)

    del dset_kwargs['max_samples']
    val_dset = LAIONSceneGraphDataset(**dset_kwargs)


    return train_dset, val_dset

def build_laion_loaders(args):

    train_dset, val_dset = build_laion_dsets(args)


    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.workers,
        'shuffle': True,
        'collate_fn': my_laion_collate_fn,

    }
    train_loader = DataLoader(train_dset, **loader_kwargs)
    train_loader.num_samples = len(train_dset)

    print('There are %d train samples' % len(train_dset))


    loader_kwargs['batch_size'] = args.val_batch_size
    loader_kwargs['shuffle'] = False

    val_loader = DataLoader(val_dset, **loader_kwargs)
    val_loader.num_samples = len(val_dset)

    print('There are %d validation samples' % len(val_dset))

    return train_loader, val_loader




