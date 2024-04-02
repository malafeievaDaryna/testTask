#pragma once

#include <vector>
#include <directxmath.h>
#include <mutex>
#include "Utils.h"

#define NOMINMAX

class BulletManager {
public:
    struct Bullet {
        DirectX::XMVECTOR pos;
        DirectX::XMVECTOR dir;
        float speed;
        float time_creation;
        float life_time;
        float distanceToOrigin; // the destination of the bullet: length(life_time * speed * dir)
        // at the first Update processing we can get to know whether the bullet intersects the walls at all and which wall it is going to intersect if yes
        bool isPrecalculationCollision{false};
        int32_t idOfTheWall{-1};  // precalculated id of the wall which is going be hit
        int32_t timeOfCollisionWithWallS{-1}; 
        Bullet(const DirectX::XMVECTOR& _pos, const DirectX::XMVECTOR& _dir, float _speed, float _time_creation, float _life_time,
               float _distanceToOrigin)
            : pos(_pos),
              dir(_dir),
              speed(_speed),
              time_creation(_time_creation),
              life_time(_life_time),
              distanceToOrigin(_distanceToOrigin) {
        }
    };

    struct BulletInstance {
        DirectX::XMFLOAT3 pos;
        BulletInstance(const DirectX::XMFLOAT3& _pos) : pos(_pos) {
        }
    };

    BulletManager(std::vector<utils::Wall>& walls);
    ~BulletManager() = default;

    void Update(float time_sec);
    void Fire(const DirectX::XMVECTOR& pos, const DirectX::XMVECTOR& dir, float speed, float time, float life_time);
    uint32_t getBulletsAmount() {
        return mBullets.size();
    }
    const std::vector<BulletInstance>& getBulletsGPUData() {
        return mBulletInstances;
    }

private:
    bool BulletManager::getTimeOfIntersection(float time_sec, const Bullet& bullet, const utils::Wall& wall,
                                              float& out_timeIntersection,
                                              DirectX::XMFLOAT3& out_intersectionPoint);

private:
    std::vector<utils::Wall>& mWalls;
    std::vector<Bullet> mBullets;
    std::vector<Bullet> mBulletsSwap; // intermediate storage
    std::mutex mSyncMngr;
    std::vector<BulletInstance> mBulletInstances;  // gpu data
};
