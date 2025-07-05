import numpy as np
from numba import njit
import time

# --- CONFIG ---
MAX_USERS = 10000
BOARD_W, BOARD_H = 1920, 1080
CHUNK_SIZE = 100

# --- DATA STRUCTURES ---

class User:
    def __init__(self, UUID, username, status, lastLogin, clientW, clientH, isMoving=False, anchor=None):
        self.UUID = UUID
        self.username = username
        self.status = status
        self.lastLogin = lastLogin
        self.clientW = clientW
        self.clientH = clientH
        self.anchor = anchor
        self.currentPos = None
        self.chunkID = None
        self.isMoving = isMoving

    def __repr__(self):
        return f"<User {self.username} at {self.currentPos}, status={self.status}>"

class Post:
    def __init__(self, postID, postType, postMedia, postTime, postedBy:User, postedOn, postSize, state):
        self.postID = postID
        self.postType = postType
        self.postMedia = postMedia
        self.postTime = postTime
        self.postedBy = postedBy
        self.postedOn = postedOn
        self.postSize = postSize
        self.state = state

    def __repr__(self):
        return f"<Post {self.postID} type={self.postType} at {self.postedOn}>"


class Board:
    def __init__(self, boardDim):
        self.boardDim = boardDim
        self.users = {}  # UUID -> User
        self.user_positions = np.full((MAX_USERS, 2), -1, dtype=np.int32)  # Preallocated
        self.uuid_to_idx = {}  # UUID -> row idx in user_positions
        self.occupied = set()  # (x, y) tuples
        self.posts = {}  # postID -> Post
        self.chunk_map = {}  # (chunk_x, chunk_y) -> set of postIDs
        self.chunkUsermap = {}
        self.next_user_idx = 0

    def add_user(self, user: User, maxTries=10):
        # Assign index
        if user.UUID in self.uuid_to_idx:
            idx = self.uuid_to_idx[user.UUID]
        else:
            idx = self.next_user_idx
            self.uuid_to_idx[user.UUID] = idx
            self.next_user_idx += 1

        # Find spawn position
        if user.anchor:
            pos = user.anchor
            if tuple(pos) in self.occupied:
                print(f"Anchor occupied for {user.username}")
                return False
        else:
            for _ in range(maxTries):
                pos = np.random.randint([0, 0], [self.boardDim[0], self.boardDim[1]])
                if tuple(pos) not in self.occupied:
                    break
            else:
                print(f"No free tile for {user.username}")
                return False

        user.currentPos = tuple(pos)
        user.chunkID = (user.currentPos[0] // CHUNK_SIZE, user.currentPos[1] // CHUNK_SIZE)
        self.users[user.UUID] = user
        self.user_positions[idx] = user.currentPos
        self.occupied.add(user.currentPos)
        self._subscribeUsertoChunk(user)
        self._updateChunk(user.chunkID)
        return True

    def move_user(self, uuid, dx, dy):
        if uuid not in self.users:
            print(f"No user {uuid}")
            return False
        user = self.users[uuid]
        idx = self.uuid_to_idx[uuid]
        old_pos = user.currentPos
        new_x = np.clip(old_pos[0] + dx, 0, self.boardDim[0] - 1)
        new_y = np.clip(old_pos[1] + dy, 0, self.boardDim[1] - 1)
        new_pos = (new_x, new_y)
        if new_pos in self.occupied:
            print(f"Target tile occupied for {user.username}")
            return False
        # Update positions
        self.occupied.remove(old_pos)
        self.occupied.add(new_pos)
        user.currentPos = new_pos
        self.user_positions[idx] = new_pos
        oldChunk = user.chunkID
        user.chunkID = (new_x // CHUNK_SIZE, new_y // CHUNK_SIZE)
        if oldChunk != user.chunkID:
            if oldChunk in self.chunkUsermap:
                self.chunkUsermap[oldChunk].discard(user.UUID)
                if not self.chunkUsermap[oldChunk]:
                    del self.chunkUsermap[oldChunk]

            self._subscribeUsertoChunk(user)

            # Update both chunks
            self._updateChunk(oldChunk)
            self._updateChunk(user.chunkID)


        return True

    def add_post(self, post: Post):
        self.posts[post.postID] = post
        # Add to chunk map (bounding box only)
        x, y = post.postedOn
        w, h = post.postSize
        chunk_x0, chunk_y0 = x // CHUNK_SIZE, y // CHUNK_SIZE
        chunk_x1, chunk_y1 = (x + w - 1) // CHUNK_SIZE, (y + h - 1) // CHUNK_SIZE
        for cx in range(chunk_x0, chunk_x1 + 1):
            for cy in range(chunk_y0, chunk_y1 + 1):
                key = (cx, cy)
                if key not in self.chunk_map:
                    self.chunk_map[key] = set()
                self.chunk_map[key].add(post.postID)
        # Mark tiles as occupied (for small posts)
        if w * h < 1000:
            for dx in range(w):
                for dy in range(h):
                    tile = (x + dx, y + dy)
                    if 0 <= tile[0] < self.boardDim[0] and 0 <= tile[1] < self.boardDim[1]:
                        self.occupied.add(tile)
        self._updateChunk(post.postedBy.chunkID)

    def posts_in_viewport(self, x0, y0, w, h):
        # Efficient chunked query for posts in viewport
        cx0, cy0 = x0 // CHUNK_SIZE, y0 // CHUNK_SIZE
        cx1, cy1 = (x0 + w - 1) // CHUNK_SIZE, (y0 + h - 1) // CHUNK_SIZE
        post_ids = set()
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                key = (cx, cy)
                if key in self.chunk_map:
                    post_ids.update(self.chunk_map[key])
        # Filter posts that actually intersect the viewport
        result = []
        for pid in post_ids:
            p = self.posts[pid]
            px, py = p.postedOn
            pw, ph = p.postSize
            if not (px + pw <= x0 or px >= x0 + w or py + ph <= y0 or py >= y0 + h):
                result.append(p)
        return result
    
    def _updateChunk(self, chunkId):
        users_in_chunk = self.chunkUsermap.get(chunkId, set())
        posts_in_chunk_ids = self.chunk_map.get(chunkId, set())
        
        posts_in_chunk = [self.posts[pid] for pid in posts_in_chunk_ids]
        
        # Now prepare a payload to send to clients
        update_payload = {
            "chunkId": chunkId,
            "users": list(users_in_chunk),
            "posts": [p.__dict__ for p in posts_in_chunk]
        }
        return update_payload

    def _subscribeUsertoChunk(self, user:User):
        chunkId = user.chunkID
        if chunkId not in self.chunkUsermap:
            self.chunkUsermap[chunkId] = set()
        self.chunkUsermap[chunkId].add(user.UUID)

    def remove_user(self, uuid):
        if uuid not in self.users:
            print(f"No user with UUID {uuid} exists.")
            return False

        user = self.users[uuid]
        idx = self.uuid_to_idx[uuid]

        # Clean up
        del self.users[uuid]
        del self.uuid_to_idx[uuid]
        self.user_positions[idx] = [-1, -1]
        self.occupied.discard(user.currentPos)

        chunkID = user.chunkID
        if chunkID in self.chunkUsermap:
            self.chunkUsermap[chunkID].discard(uuid)
            if not self.chunkUsermap[chunkID]:
                del self.chunkUsermap[chunkID]

        self._updateChunk(chunkID)

        print(f"User {user.username} removed from board.")
        return True


    def __repr__(self):
        return f"<Board Users={len(self.users)}, Posts={len(self.posts)}, Chunks={len(self.chunk_map)}>"



