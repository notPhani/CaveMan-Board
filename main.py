import numpy as np
from numba import njit
import time

# === Classes from your code ===
@njit
def position_exists(userPositions, pos):
    for i in range(userPositions.shape[0]):
        if userPositions[i, 0] == pos[0] and userPositions[i, 1] == pos[1]:
            return True
    return False

class User:
    def __init__(self, UUID, username, status, lastLogin, clientW, clientH, anchor=None, currentPos=None, currentChunk=None):
        self.UUID = UUID
        self.username = username
        self.status = status
        self.lastLogin = lastLogin
        self.clientW = clientW
        self.clientH = clientH
        self.anchor = anchor
        self.currentPos = currentPos
        self.currentChunk = currentChunk
        self.chunkID = None

    def __repr__(self):
        return f"<User {self.username} at {self.currentPos}, status={self.status}>"

class Post:
    def __init__(self, postID, postType, postMedia, postTime, postedBy, postedOn, postSize, state):
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
    def __init__(self, users:list, boardDim: tuple, state: str):
        self.users = users
        self.boardDim = boardDim
        self.state = state
        self.posts = []
        self.post_bitmap = {}

        if users:
            self.userPositions = np.stack([u.currentPos for u in users if u.currentPos is not None])
            self.numUsersOnline = sum(u.status != "dead" for u in users)
        else:
            self.userPositions = np.empty((0, 2), dtype=int)
            self.numUsersOnline = 0

    def add_user(self, user: User, maxTries: int, isolationRadius: int, chunkSize=None):
        if user.anchor:
            user.currentPos = user.anchor
        else:
            for _ in range(maxTries):
                tempPos = np.random.randint(
                    low=[0, 0],
                    high=[self.boardDim[0], self.boardDim[1]]
                )
                if not position_exists(self.userPositions,tuple(tempPos)):
                    user.currentPos = tuple(tempPos)
                    break
            else:
                print(f"Failed to find free tile for user {user.username}. Leaving position None.")

        # Compute initial chunkID if chunking is used
        if chunkSize and user.currentPos is not None:
            chunkX = user.currentPos[0] // chunkSize[0]
            chunkY = user.currentPos[1] // chunkSize[1]
            user.chunkID = (chunkX, chunkY)
        else:
            user.chunkID = None

        self.users.append(user)
        self.userPositions = np.stack([
            u.currentPos for u in self.users if u.currentPos is not None
        ])
        self.numUsersOnline = sum(u.status != "dead" for u in self.users)

    @njit
    def _position_exists(userPositions, pos):
        for i in range(userPositions.shape[0]):
            if userPositions[i, 0] == pos[0] and userPositions[i, 1] == pos[1]:
                return True
        return False


    def move_user(self, uuid, dragX, dragY, chunkSize=None):
        user = next((u for u in self.users if u.UUID == uuid), None)
        if user is None:
            print(f"No user with UUID {uuid} found!")
            return None

        if user.currentPos is None:
            print(f"User {user.username} has no current position.")
            return None

        newX = int(user.currentPos[0] + dragX)
        newY = int(user.currentPos[1] + dragY)

        newX = np.clip(newX, 0, self.boardDim[0] - 1)
        newY = np.clip(newY, 0, self.boardDim[1] - 1)

        oldChunkID = getattr(user, "chunkID", None)
        newChunkID = oldChunkID

        if chunkSize:
            newChunkX = newX // chunkSize[0]
            newChunkY = newY // chunkSize[1]
            newChunkID = (newChunkX, newChunkY)

            if newChunkID != oldChunkID:
                print(f"User {user.username} moved chunk from {oldChunkID} → {newChunkID}")
                user.chunkID = newChunkID

        user.currentPos = (newX, newY)

        self.userPositions = np.stack([
            u.currentPos for u in self.users if u.currentPos is not None
        ])

        if newChunkID != oldChunkID:
            return newChunkID
        else:
            return None

    def add_post(self, post: Post):
        self.posts.append(post)
        x_start, y_start = post.postedOn
        w, h = post.postSize

        for x in range(x_start, min(x_start + w, self.boardDim[0])):
            for y in range(y_start, min(y_start + h, self.boardDim[1])):
                tile = (x, y)
                if tile not in self.post_bitmap:
                    self.post_bitmap[tile] = []
                self.post_bitmap[tile].append(post.postID)

    def __repr__(self):
        return f"<Board Users={len(self.users)} Online={self.numUsersOnline}, Posts={len(self.posts)}>"

# === BENCHMARK TEST ===

if __name__ == "__main__":
    print("\n=== Caveman Board Benchmark ===")

    # Create giant board
    board_size = (1920, 1080)
    board = Board([], board_size, "running")

    # Add users
    start = time.time()
    for i in range(1000):
        u = User(
            UUID=f"uuid-{i}",
            username=f"user{i}",
            status="alive",
            lastLogin="2025-07-03",
            clientW=800,
            clientH=600
        )
        board.add_user(u, maxTries=10, isolationRadius=50)
    end = time.time()
    print(f"Added 1000 users in {end-start:.4f} seconds")

    # Move users
    start = time.time()
    for u in board.users:
        board.move_user(u.UUID, dragX=5, dragY=-3, chunkSize=(100,100))
    end = time.time()
    print(f"Moved 1000 users in {end-start:.4f} seconds")

    # Add small posts
    start = time.time()
    for i in range(500):
        p = Post(
            postID=f"post-{i}",
            postType="text",
            postMedia=f"post_{i}.txt",
            postTime="2025-07-03",
            postedBy=f"uuid-{i%1000}",
            postedOn=(np.random.randint(0, 1920), np.random.randint(0, 1080)),
            postSize=(5, 5),
            state="wet"
        )
        board.add_post(p)
    end = time.time()
    print(f"Added 500 small posts in {end-start:.4f} seconds")

    # Add a massive post
    start = time.time()
    big_post = Post(
        postID="big-one",
        postType="image",
        postMedia="huge_image.png",
        postTime="2025-07-03",
        postedBy="uuid-0",
        postedOn=(0,0),
        postSize=(1920, 1080),
        state="wet"
    )
    board.add_post(big_post)
    end = time.time()
    print(f"Added huge post (1920x1080) in {end-start:.4f} seconds")

    print("Board summary:", board)
    print("Sample tile posts at (10,10):", board.post_bitmap.get((10,10)))
