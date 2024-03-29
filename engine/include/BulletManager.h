#pragma once

#include <vector>
#include <directxmath.h>
#include "Utils.h"

class BulletManager {
    struct Bullet {
        DirectX::XMVECTOR pos;
        DirectX::XMVECTOR dir;
        float speed;
        float time_creation;
        float life_time;
        Bullet(const DirectX::XMVECTOR& _pos, const DirectX::XMVECTOR& _dir, float _speed, float _time_creation, float _life_time)
            : pos(_pos), dir(_dir), speed(_speed), time_creation(_time_creation), life_time(_life_time) {
        }
    };

public:
    BulletManager(std::vector<utils::Wall>& walls);
    ~BulletManager() = default;

    void Update(float time_sec);
    void Fire(const DirectX::XMFLOAT3& pos, const DirectX::XMFLOAT3& dir, float speed, float time, float life_time);

private:
    std::vector<utils::Wall>& mWalls;
    std::vector<Bullet> mBullets;
};
