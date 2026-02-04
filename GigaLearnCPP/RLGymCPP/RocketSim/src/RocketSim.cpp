#include "RocketSim.h"

#include "../libsrc/bullet3-3.24/BulletCollision/CollisionDispatch/btInternalEdgeUtility.h"
#include "../libsrc/bullet3-3.24/BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h"
#include "../libsrc/bullet3-3.24/BulletCollision/CollisionShapes/btTriangleMesh.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

using namespace RocketSim;

std::filesystem::path RocketSim::_collisionMeshesFolder = {};
std::mutex RocketSim::_beginInitMutex = {};

struct MeshHashSet {
  std::unordered_map<uint32_t, int> hashes;
  void AddAll(std::initializer_list<uint32_t> hashesToAdd) {
    for (uint32_t hash : hashesToAdd)
      hashes[hash] = 0;
  }

  MeshHashSet(GameMode gameMode) {
    if (gameMode == GameMode::SOCCAR) {
      AddAll({0xA160BAF9, 0x2811EEE8, 0xB81AC8B9, 0x760358D3, 0x73AE4940,
              0x918F4A4E, 0x1F8EE550, 0x255BA8C1, 0x14B84668, 0xEC759EBF,
              0x94FB0D5C, 0xDEA07102, 0xBD4FBEA8, 0x39A47F63, 0x3D79D25D,
              0xD84C7A68});
    } else if (gameMode == GameMode::HOOPS) {
      AddAll({0x72F2359E, 0x5ED14A26, 0XFD5A0D07, 0x92AFA5B5, 0x0E4133C7,
              0x399E8B5F, 0XBB9D4FB5, 0x8C87FB93, 0x1CFD0E16, 0xE19E1DF6,
              0x9CA179DC, 0x16F3CC19});
    }
  }

  int &operator[](uint32_t hash) { return hashes[hash]; }
};

static RocketSimStage stage = RocketSimStage::UNINITIALIZED;
RocketSimStage RocketSim::GetStage() { return stage; }

std::vector<btBvhTriangleMeshShape *> &
RocketSim::GetArenaCollisionShapes(GameMode gameMode) {
  static std::map<GameMode, std::vector<btBvhTriangleMeshShape *>>
      arenaCollisionMeshes;
  return arenaCollisionMeshes[gameMode];
}

namespace {
constexpr uint32_t kMeshCacheMagic = 0x524D4348; // "RMCH"
constexpr uint32_t kMeshCacheVersion = 2;

int64_t GetMaxMeshWriteTime(const std::filesystem::path &folder) {
  if (!std::filesystem::exists(folder))
    return -1;
  int64_t maxTime = -1;
  for (auto &entry : std::filesystem::directory_iterator(folder)) {
    auto entryPath = entry.path();
    if (entryPath.has_extension() &&
        entryPath.extension() == COLLISION_MESH_FILE_EXTENSION) {
      auto ftime = std::filesystem::last_write_time(entryPath);
      auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    ftime.time_since_epoch())
                    .count();
      if (ns > maxTime)
        maxTime = ns;
    }
  }
  return maxTime;
}

bool LoadMeshCache(const std::filesystem::path &cachePath,
                   int64_t expectedMaxWriteTime,
                   std::vector<CachedMeshData> &outMeshes) {
  if (!std::filesystem::exists(cachePath))
    return false;
  std::ifstream in(cachePath, std::ios::binary);
  if (!in.good())
    return false;

  uint32_t magic = 0;
  uint32_t version = 0;
  int64_t cachedMaxWriteTime = -1;
  uint32_t fileCount = 0;

  in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char *>(&version), sizeof(version));
  in.read(reinterpret_cast<char *>(&cachedMaxWriteTime),
          sizeof(cachedMaxWriteTime));
  in.read(reinterpret_cast<char *>(&fileCount), sizeof(fileCount));

  if (!in.good() || magic != kMeshCacheMagic)
    return false;

  // We still support version 1 for transition, but we'll re-save as version 2
  if (version != 1 && version != 2)
    return false;

  if (expectedMaxWriteTime > cachedMaxWriteTime)
    return false;

  outMeshes.clear();
  outMeshes.reserve(fileCount);
  for (uint32_t i = 0; i < fileCount; i++) {
    CachedMeshData mesh;

    auto ReadData = [&](FileData &data) {
      uint32_t size = 0;
      in.read(reinterpret_cast<char *>(&size), sizeof(size));
      if (!in.good())
        return false;
      data.resize(size);
      if (size > 0) {
        in.read(reinterpret_cast<char *>(data.data()), size);
        if (!in.good())
          return false;
      }
      return true;
    };

    if (!ReadData(mesh.rawData))
      return false;

    if (version >= 2) {
      if (!ReadData(mesh.bvhData))
        return false;
      if (!ReadData(mesh.infoMapData))
        return false;
    }

    outMeshes.push_back(std::move(mesh));
  }

  return true;
}

void SaveMeshCache(const std::filesystem::path &cachePath, int64_t maxWriteTime,
                   const std::vector<CachedMeshData> &meshes) {
  std::ofstream out(cachePath, std::ios::binary | std::ios::trunc);
  if (!out.good())
    return;

  uint32_t magic = kMeshCacheMagic;
  uint32_t version = kMeshCacheVersion;
  uint32_t fileCount = static_cast<uint32_t>(meshes.size());

  out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  out.write(reinterpret_cast<const char *>(&version), sizeof(version));
  out.write(reinterpret_cast<const char *>(&maxWriteTime),
            sizeof(maxWriteTime));
  out.write(reinterpret_cast<const char *>(&fileCount), sizeof(fileCount));

  for (const auto &mesh : meshes) {
    auto WriteData = [&](const FileData &data) {
      uint32_t size = static_cast<uint32_t>(data.size());
      out.write(reinterpret_cast<const char *>(&size), sizeof(size));
      if (size > 0)
        out.write(reinterpret_cast<const char *>(data.data()), size);
    };

    WriteData(mesh.rawData);
    WriteData(mesh.bvhData);
    WriteData(mesh.infoMapData);
  }
}
} // namespace

void RocketSim::Init(std::filesystem::path collisionMeshesFolder, bool silent) {
  std::map<GameMode, std::vector<CachedMeshData>> meshFileMap = {};

  constexpr GameMode GAMEMODES_WITH_UNIQUE_MESHES[] = {
      GameMode::SOCCAR,
      GameMode::HOOPS,
      GameMode::DROPSHOT,
  };

  bool cacheUpdated = false;

  for (GameMode gameMode :
       GAMEMODES_WITH_UNIQUE_MESHES) { // Load collision meshes for soccar and
                                       // hoops
    std::filesystem::path basePath = collisionMeshesFolder;
    std::filesystem::path soccarMeshesFolder =
        basePath / GAMEMODE_STRS[(int)gameMode];

    if (!std::filesystem::exists(soccarMeshesFolder))
      continue;

    auto maxWriteTime = GetMaxMeshWriteTime(soccarMeshesFolder);
    std::filesystem::path cachePath =
        soccarMeshesFolder / "collision_meshes.cache";
    bool loadedFromCache =
        LoadMeshCache(cachePath, maxWriteTime, meshFileMap[gameMode]);

    if (!loadedFromCache) {
      // Load collision meshes raw
      auto dirItr = std::filesystem::directory_iterator(soccarMeshesFolder);
      for (auto &entry : dirItr) {
        auto entryPath = entry.path();
        if (entryPath.has_extension() &&
            entryPath.extension() == COLLISION_MESH_FILE_EXTENSION) {
          DataStreamIn streamIn = DataStreamIn(entryPath, false);
          CachedMeshData mesh;
          mesh.rawData = streamIn.data;
          meshFileMap[gameMode].push_back(std::move(mesh));
        }
      }
      cacheUpdated = true;
    } else {
      // Check if we need to upgrade version 1 cache
      for (auto &mesh : meshFileMap[gameMode]) {
        if (mesh.bvhData.empty()) {
          cacheUpdated = true;
          break;
        }
      }
    }
  }

  RocketSim::InitFromMemCached(meshFileMap, silent);

  if (cacheUpdated) {
    for (GameMode gameMode : GAMEMODES_WITH_UNIQUE_MESHES) {
      std::filesystem::path basePath = collisionMeshesFolder;
      std::filesystem::path soccarMeshesFolder =
          basePath / GAMEMODE_STRS[(int)gameMode];
      if (!std::filesystem::exists(soccarMeshesFolder))
        continue;

      auto maxWriteTime = GetMaxMeshWriteTime(soccarMeshesFolder);
      std::filesystem::path cachePath =
          soccarMeshesFolder / "collision_meshes.cache";
      SaveMeshCache(cachePath, maxWriteTime, meshFileMap[gameMode]);
    }
  }

  _collisionMeshesFolder = collisionMeshesFolder;
}

void RocketSim::InitFromMem(
    const std::map<GameMode, std::vector<FileData>> &meshFilesMap,
    bool silent) {
  std::map<GameMode, std::vector<CachedMeshData>> cachedMeshesMap;
  for (auto &pair : meshFilesMap) {
    for (auto &fileData : pair.second) {
      CachedMeshData mesh;
      mesh.rawData = fileData;
      cachedMeshesMap[pair.first].push_back(std::move(mesh));
    }
  }
  InitFromMemCached(cachedMeshesMap, silent);
}

void RocketSim::InitFromMemCached(
    const std::map<GameMode, std::vector<CachedMeshData>> &meshFilesMap,
    bool silent) {

  constexpr char MSG_PREFIX[] = "RocketSim::Init(): ";

  _collisionMeshesFolder = "<MESH FILES LOADED FROM MEMORY>";

  _beginInitMutex.lock();
  {
    if (stage != RocketSimStage::UNINITIALIZED) {
      if (!silent)
        RS_WARN("RocketSim::Init() called again after already initialized, "
                "ignoring...");
      _beginInitMutex.unlock();
      return;
    }

    if (!silent)
      RS_LOG("Initializing RocketSim version " RS_VERSION
             ", created by ZealanL...");

    stage = RocketSimStage::INITIALIZING;

    uint64_t startMS = RS_CUR_MS();

    // Init dropshot stuff
    DropshotTiles::Init();

    for (auto &mapPair : meshFilesMap) {
      GameMode gameMode = mapPair.first;
      auto &meshFiles = mapPair.second;

      if (!silent)
        RS_LOG("Loading arena meshes for " << GAMEMODE_STRS[(int)gameMode]
                                           << "...");

      if (meshFiles.empty()) {
        if (!silent)
          RS_LOG(" > No meshes, skipping");
        continue;
      }

      auto &meshes = GetArenaCollisionShapes(gameMode);

      MeshHashSet targetHashes = MeshHashSet(gameMode);

      // Load collision meshes
      int idx = 0;
      for (auto &entry : const_cast<std::vector<CachedMeshData> &>(meshFiles)) {
        DataStreamIn dataStream = {};
        dataStream.data = entry.rawData;
        CollisionMeshFile meshFile = {};
        meshFile.ReadFromStream(dataStream, silent);
        int &hashCount = targetHashes[meshFile.hash];

        if (hashCount > 0) {
          if (!silent)
            RS_WARN(MSG_PREFIX << "Collision mesh [" << idx
                               << "] is a duplicate (0x" << std::hex
                               << meshFile.hash << "), "
                               << "already loaded a mesh with the same hash.");
        } else if (targetHashes.hashes.count(meshFile.hash) == 0) {
          if (!silent)
            RS_WARN(MSG_PREFIX << "Collision mesh [" << idx
                               << "] does not match any known "
                               << GAMEMODE_STRS[(int)gameMode]
                               << " collision mesh (0x" << std::hex
                               << meshFile.hash << "), "
                               << "make sure they were dumped from a normal "
                               << GAMEMODE_STRS[(int)gameMode] << " arena.");
        }
        hashCount++;

        btTriangleMesh *triMesh = meshFile.MakeBulletMesh();

        bool hasCachedBvh = !entry.bvhData.empty();
        bool hasCachedEdges = !entry.infoMapData.empty();

        auto bvtMesh = new btBvhTriangleMeshShape(
            triMesh, false); // Don't build BVH here if we have cache

        if (hasCachedBvh) {
          bvtMesh->getOptimizedBvh()->deSerialize(
              entry.bvhData.data(), (unsigned)entry.bvhData.size(), false);
        } else {
          bvtMesh->buildOptimizedBvh();
          // Serialize back to entry for saving to cache later
          entry.bvhData.resize(
              bvtMesh->getOptimizedBvh()->getSerializationSize());
          bvtMesh->getOptimizedBvh()->serialize(
              entry.bvhData.data(), (unsigned)entry.bvhData.size(), false);
        }

        btTriangleInfoMap *infoMap = new btTriangleInfoMap();
        if (hasCachedEdges) {
          infoMap->deSerialize(entry.infoMapData.data(),
                               (unsigned)entry.infoMapData.size(), false);
        } else {
          btGenerateInternalEdgeInfo(bvtMesh, infoMap);
          // Serialize back to entry
          entry.infoMapData.resize(infoMap->getSerializationSize());
          infoMap->serialize(entry.infoMapData.data(),
                             (unsigned)entry.infoMapData.size(), false);
        }

        bvtMesh->setTriangleInfoMap(infoMap);
        meshes.push_back(bvtMesh);

        idx++;
      }
    }

    if (!silent) {
      RS_LOG(MSG_PREFIX << "Finished loading arena collision meshes:");
      RS_LOG(" > Soccar: " << GetArenaCollisionShapes(GameMode::SOCCAR).size());
      RS_LOG(" > Hoops: " << GetArenaCollisionShapes(GameMode::HOOPS).size());
      RS_LOG(" > Dropshot: "
             << GetArenaCollisionShapes(GameMode::DROPSHOT).size());
    }

    uint64_t elapsedMS = RS_CUR_MS() - startMS;

    if (!silent)
      RS_LOG("Finished initializing RocketSim in " << (elapsedMS / 1000.f)
                                                   << "s!");

    stage = RocketSimStage::INITIALIZED;
  }
  _beginInitMutex.unlock();
}

void RocketSim::AssertInitialized(const char *errorMsgPrefix) {
  if (stage != RocketSimStage::INITIALIZED) {
    RS_ERR_CLOSE(
        errorMsgPrefix
        << "RocketSim has not been initialized, call RocketSim::Init() first")
  }
}
