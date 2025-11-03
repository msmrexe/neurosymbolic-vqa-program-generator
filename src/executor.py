import json
import random
from typing import List, Dict, Any, Optional

from src.utils.scene_utils import load_clevr_scenes
from src.vocabulary import load_vocab
from src.utils.logger import log

# --- CLEVR Constants ---
CLEVR_COLORS = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
CLEVR_MATERIALS = ['rubber', 'metal']
CLEVR_SHAPES = ['cube', 'cylinder', 'sphere']
CLEVR_SIZES = ['large', 'small']

CLEVR_ANSWER_CANDIDATES = {
    'count': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'equal_color': ['yes', 'no'],
    'equal_integer': ['yes', 'no'],
    'equal_material': ['yes', 'no'],
    'equal_shape': ['yes', 'no'],
    'equal_size': ['yes', 'no'],
    'exist': ['yes', 'no'],
    'greater_than': ['yes', 'no'],
    'less_than': ['yes', 'no'],
    'query_color': CLEVR_COLORS,
    'query_material': CLEVR_MATERIALS,
    'query_size': CLEVR_SIZES,
    'query_shape': CLEVR_SHAPES,
    'same_color': ['yes', 'no'],
    'same_material': ['yes', 'no'],
    'same_size': ['yes', 'no'],
    'same_shape': ['yes', 'no']
}

# Type definition for a scene object
SceneObject = Dict[str, Any]


class ClevrExecutor:
    """
    Symbolic program executor for CLEVR.
    This executor interprets a program in POSTFIX (Reverse Polish) notation.
    It uses a stack-based approach to evaluate the program.
    """

    def __init__(self, train_scene_json: str, val_scene_json: str, vocab_json: str):
        log.info("Initializing ClevrExecutor...")
        self.scenes = {
            'train': load_clevr_scenes(train_scene_json),
            'val': load_clevr_scenes(val_scene_json)
        }
        self.vocab = load_vocab(vocab_json)
        self.idx_to_token = self.vocab['program_idx_to_token']
        
        self.execution_trace = []
        self._modules = {}
        self._register_modules()
        log.info("ClevrExecutor initialized with registered modules.")

    def run(self, program_indices: List[int], image_index: int, split: str, 
            guess: bool = False, debug: bool = False) -> str:
        """
        Executes a tokenized program sequence on a specific scene.

        Args:
            program_indices (List[int]): The tokenized program in postfix order.
            image_index (int): The index of the scene to use.
            split (str): The dataset split ('train' or 'val').
            guess (bool): If true, guess a random answer on error.
            debug (bool): If true, print execution trace.

        Returns:
            str: The final answer, or 'error'.
        """
        assert split in ['train', 'val'], "Split must be 'train' or 'val'"
        
        # Find the <END> token
        end_token_idx = self.vocab['program_token_to_idx']['<END>']
        try:
            end_pos = program_indices.index(end_token_idx)
        except ValueError:
            # No <END> token, try to execute the whole sequence
            end_pos = len(program_indices)

        # We only execute up to (but not including) the <END> token
        program_to_run = program_indices[:end_pos]
        
        if not program_to_run:
            return 'error'
            
        scene = self.scenes[split][image_index]
        self.execution_trace = []
        
        # This is a stack-based machine.
        stack: List[Any] = []

        # Iterate through the program in POSTFIX order (i.e., forward)
        # The original code iterated backward, which implies the program
        # was stored in reverse. We assume a standard forward-iterating
        # postfix evaluation.
        # NOTE: Re-reading the *original* code, it iterates from (end-1) down to 0.
        # This means the program *was* stored in prefix and evaluated backward,
        # or stored in postfix and evaluated backward (which makes no sense).
        
        # Let's follow the original's backward iteration.
        # This means program is [token1, token2, 'scene', '<END>']
        # And it's evaluated as:
        # 1. 'scene' (push scene)
        # 2. 'token2' (pop scene, push result)
        # 3. 'token1' (pop result, push final)
        # This is POSTFIX. The original code just iterates it backward.
        
        # We will iterate backward from the token *before* <END>
        for i in range(end_pos - 1, -1, -1):
            token = self.idx_to_token.get(program_indices[i])
            if token is None:
                return 'error'

            if debug:
                log.debug(f"Executing token: {token}")

            if token == 'scene':
                stack.append(list(scene)) # Push the scene objects
            elif token in self._modules:
                module = self._modules[token]
                
                # Determine arity (how many args the function needs)
                is_relate_or_same = token.startswith('relate') or token.startswith('same')
                arity = 1 if is_relate_or_same else module.__code__.co_argcount - 1 # -1 for 'self'
                
                if arity == 0:
                    return 'error' # Should not happen
                elif arity == 1:
                    if not stack:
                        return 'error' # Stack underflow
                    arg1 = stack.pop()
                    result = module(arg1)
                    stack.append(result)
                elif arity == 2:
                    if len(stack) < 2:
                        return 'error' # Stack underflow
                    # Note: order is important!
                    arg2 = stack.pop()
                    arg1 = stack.pop()
                    if is_relate_or_same:
                        # Relate/Same functions need the full scene as the 2nd arg
                        result = module(arg1, scene)
                    else:
                        result = module(arg1, arg2)
                    stack.append(result)

                if result == 'error':
                    return 'error'
            
            self.execution_trace.append(stack[-1] if stack else None)
            if debug:
                log.debug(f"  Stack top: {self._object_info(stack[-1]) if stack else 'EMPTY'}")

        # The final answer should be the only item on the stack
        if len(stack) != 1:
            return 'error'
            
        final_answer = str(stack[0])

        if final_answer == 'error' and guess:
            # Guess a random answer
            final_token = self.idx_to_token.get(program_indices[0]) # First token is final op
            if final_token in CLEVR_ANSWER_CANDIDATES:
                return random.choice(CLEVR_ANSWER_CANDIDATES[final_token])
        
        return final_answer

    def _print_debug_message(self, x: Any):
        log.debug(self._object_info(x))

    def _object_info(self, x: Any) -> str:
        if isinstance(x, list):
            return f"List of {len(x)} items: {[self._object_info(o) for o in x[:3]]}..."
        if isinstance(x, dict) and 'shape' in x:
            return f"{x['size']} {x['color']} {x['material']} {x['shape']}"
        return str(x)
    
    # --- Module Registration ---
    
    def _register_modules(self):
        self._modules['count'] = self._count
        self._modules['equal_color'] = self._equal_color
        self._modules['equal_integer'] = self._equal_integer
        self._modules['equal_material'] = self._equal_material
        self._modules['equal_shape'] = self._equal_shape
        self._modules['equal_size'] = self._equal_size
        self._modules['exist'] = self._exist
        self._modules['filter_color[blue]'] = self._filter_blue
        self._modules['filter_color[brown]'] = self._filter_brown
        self._modules['filter_color[cyan]'] = self._filter_cyan
        self._modules['filter_color[gray]'] = self._filter_gray
        self._modules['filter_color[green]'] = self._filter_green
        self._modules['filter_color[purple]'] = self._filter_purple
        self._modules['filter_color[red]'] = self._filter_red
        self._modules['filter_color[yellow]'] = self._filter_yellow
        self._modules['filter_material[rubber]'] = self._filter_rubber
        self._modules['filter_material[metal]'] = self._filter_metal
        self._modules['filter_shape[cube]'] = self._filter_cube
        self._modules['filter_shape[cylinder]'] = self._filter_cylinder
        self._modules['filter_shape[sphere]'] = self._filter_sphere
        self._modules['filter_size[large]'] = self._filter_large
        self._modules['filter_size[small]'] = self._filter_small
        self._modules['greater_than'] = self._greater_than
        self._modules['less_than'] = self._less_than
        self._modules['intersect'] = self._intersect
        self._modules['query_color'] = self._query_color
        self._modules['query_material'] = self._query_material
        self._modules['query_shape'] = self._query_shape
        self._modules['query_size'] = self._query_size
        self._modules['relate[behind]'] = self._relate_behind
        self._modules['relate[front]'] = self._relate_front
        self._modules['relate[left]'] = self._relate_left
        self._modules['relate[right]'] = self._relate_right
        self._modules['same_color'] = self._same_color
        self._modules['same_material'] = self._same_material
        self._modules['same_shape'] = self._same_shape
        self._modules['same_size'] = self._same_size
        self._modules['union'] = self._union
        self._modules['unique'] = self._unique

    # --- Symbolic Function Implementations ---
    
    def _count(self, scene: List[SceneObject]) -> Any:
        return len(scene) if isinstance(scene, list) else 'error'

    def _equal_color(self, color1: str, color2: str) -> str:
        if color1 in CLEVR_COLORS and color2 in CLEVR_COLORS:
            return 'yes' if color1 == color2 else 'no'
        return 'error'

    def _equal_integer(self, int1: int, int2: int) -> str:
        if isinstance(int1, int) and isinstance(int2, int):
            return 'yes' if int1 == int2 else 'no'
        return 'error'

    def _equal_material(self, mat1: str, mat2: str) -> str:
        if mat1 in CLEVR_MATERIALS and mat2 in CLEVR_MATERIALS:
            return 'yes' if mat1 == mat2 else 'no'
        return 'error'

    def _equal_shape(self, shape1: str, shape2: str) -> str:
        if shape1 in CLEVR_SHAPES and shape2 in CLEVR_SHAPES:
            return 'yes' if shape1 == shape2 else 'no'
        return 'error'

    def _equal_size(self, size1: str, size2: str) -> str:
        if size1 in CLEVR_SIZES and size2 in CLEVR_SIZES:
            return 'yes' if size1 == size2 else 'no'
        return 'error'

    def _exist(self, scene: List[SceneObject]) -> str:
        return 'yes' if isinstance(scene, list) and len(scene) > 0 else 'no'

    def _filter_blue(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['color'] == 'blue'] if isinstance(scene, list) else 'error'
    
    def _filter_brown(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['color'] == 'brown'] if isinstance(scene, list) else 'error'

    def _filter_cyan(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['color'] == 'cyan'] if isinstance(scene, list) else 'error'

    def _filter_gray(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['color'] == 'gray'] if isinstance(scene, list) else 'error'

    def _filter_green(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['color'] == 'green'] if isinstance(scene, list) else 'error'

    def _filter_purple(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['color'] == 'purple'] if isinstance(scene, list) else 'error'

    def _filter_red(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['color'] == 'red'] if isinstance(scene, list) else 'error'

    def _filter_yellow(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['color'] == 'yellow'] if isinstance(scene, list) else 'error'

    def _filter_rubber(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['material'] == 'rubber'] if isinstance(scene, list) else 'error'

    def _filter_metal(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['material'] == 'metal'] if isinstance(scene, list) else 'error'

    def _filter_cube(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['shape'] == 'cube'] if isinstance(scene, list) else 'error'

    def _filter_cylinder(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['shape'] == 'cylinder'] if isinstance(scene, list) else 'error'

    def _filter_sphere(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['shape'] == 'sphere'] if isinstance(scene, list) else 'error'

    def _filter_large(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['size'] == 'large'] if isinstance(scene, list) else 'error'

    def _filter_small(self, scene: List[SceneObject]) -> Any:
        return [o for o in scene if o['size'] == 'small'] if isinstance(scene, list) else 'error'

    def _greater_than(self, int1: int, int2: int) -> str:
        if isinstance(int1, int) and isinstance(int2, int):
            return 'yes' if int1 > int2 else 'no'
        return 'error'

    def _less_than(self, int1: int, int2: int) -> str:
        if isinstance(int1, int) and isinstance(int2, int):
            return 'yes' if int1 < int2 else 'no'
        return 'error'

    def _intersect(self, scene1: List[SceneObject], scene2: List[SceneObject]) -> Any:
        if isinstance(scene1, list) and isinstance(scene2, list):
            # Use object IDs for stable intersection
            ids1 = {o['id'] for o in scene1}
            return [o for o in scene2 if o['id'] in ids1]
        return 'error'

    def _query_color(self, obj: SceneObject) -> str:
        return obj.get('color', 'error')

    def _query_material(self, obj: SceneObject) -> str:
        return obj.get('material', 'error')

    def _query_shape(self, obj: SceneObject) -> str:
        return obj.get('shape', 'error')

    def _query_size(self, obj: SceneObject) -> str:
        return obj.get('size', 'error')

    def _relate_behind(self, obj: SceneObject, scene: List[SceneObject]) -> Any:
        if 'position' in obj and isinstance(scene, list):
            obj_pos = obj['position']
            return [o for o in scene if o['position'][1] < obj_pos[1]]
        return 'error'

    def _relate_front(self, obj: SceneObject, scene: List[SceneObject]) -> Any:
        if 'position' in obj and isinstance(scene, list):
            obj_pos = obj['position']
            return [o for o in scene if o['position'][1] > obj_pos[1]]
        return 'error'

    def _relate_left(self, obj: SceneObject, scene: List[SceneObject]) -> Any:
        if 'position' in obj and isinstance(scene, list):
            obj_pos = obj['position']
            return [o for o in scene if o['position'][0] < obj_pos[0]]
        return 'error'

    def _relate_right(self, obj: SceneObject, scene: List[SceneObject]) -> Any:
        if 'position' in obj and isinstance(scene, list):
            obj_pos = obj['position']
            return [o for o in scene if o['position'][0] > obj_pos[0]]
        return 'error'

    def _same_color(self, obj: SceneObject, scene: List[SceneObject]) -> Any:
        if 'color' in obj and isinstance(scene, list):
            obj_color = obj['color']
            obj_id = obj['id']
            return [o for o in scene if o['color'] == obj_color and o['id'] != obj_id]
        return 'error'

    def _same_material(self, obj: SceneObject, scene: List[SceneObject]) -> Any:
        if 'material' in obj and isinstance(scene, list):
            obj_mat = obj['material']
            obj_id = obj['id']
            return [o for o in scene if o['material'] == obj_mat and o['id'] != obj_id]
        return 'error'

    def _same_shape(self, obj: SceneObject, scene: List[SceneObject]) -> Any:
        if 'shape' in obj and isinstance(scene, list):
            obj_shape = obj['shape']
            obj_id = obj['id']
            return [o for o in scene if o['shape'] == obj_shape and o['id'] != obj_id]
        return 'error'

    def _same_size(self, obj: SceneObject, scene: List[SceneObject]) -> Any:
        if 'size' in obj and isinstance(scene, list):
            obj_size = obj['size']
            obj_id = obj['id']
            return [o for o in scene if o['size'] == obj_size and o['id'] != obj_id]
        return 'error'

    def _union(self, scene1: List[SceneObject], scene2: List[SceneObject]) -> Any:
        if isinstance(scene1, list) and isinstance(scene2, list):
            # Use object IDs for stable union
            ids1 = {o['id'] for o in scene1}
            union_list = list(scene1)
            union_list.extend(o for o in scene2 if o['id'] not in ids1)
            return union_list
        return 'error'

    def _unique(self, scene: List[SceneObject]) -> Any:
        if isinstance(scene, list) and len(scene) > 0:
            return scene[0]
        return 'error'
