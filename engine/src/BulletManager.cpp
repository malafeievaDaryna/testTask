#include "BulletManager.h"

using namespace DirectX;

BulletManager::BulletManager(std::vector<utils::Wall>& walls) : mWalls(walls) {
    mBullets.reserve(1500000); // we can have at most ~1 million bullets at once
}

void BulletManager::Update(float time_sec) {
    utils::log_debug("cur time s", time_sec);
    auto& wall = mWalls[0];
    auto& bullet = mBullets[0];
    DirectX::XMVECTOR N = wall.normal;
    DirectX::XMVECTOR P = bullet.pos;
    DirectX::XMVECTOR V = bullet.dir;
    float D = wall.distanceToOrigin;
    
    float denom = DirectX::XMVectorGetX(DirectX::XMVector3Dot(N, V));

    // Prevent divide by zero:
    if (abs(denom) <= 1e-4f) {
        utils::log_debug("denom small");
        return;
    }


    //float t = -(dot(n, p) + d) / dot(n, v);
    float t = -(DirectX::XMVectorGetX(DirectX::XMVector3Dot(N, P)) + D) / denom;

    // Use pointy end of the ray.
    // It is technically correct to compare t < 0,
    // but that may be undesirable in a raytracer.
    if (t <= 1e-4) {
        utils::log_debug("denom small");
        return;    
    }

    XMFLOAT3 intersectionPoint;
    XMStoreFloat3(&intersectionPoint, XMVectorAdd(P, t * V));
    utils::log_info("intersection happens at point", intersectionPoint.x, intersectionPoint.y, intersectionPoint.z, " time ", t);
}

void BulletManager::Fire(const DirectX::XMFLOAT3& pos, const DirectX::XMFLOAT3& dir, float speed, float time, float life_time) {
    mBullets.emplace_back(XMLoadFloat3(&pos), XMLoadFloat3(&dir), speed, time, life_time);
}
