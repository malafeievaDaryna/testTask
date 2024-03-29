#include "BulletManager.h"

BulletManager::BulletManager() {
    mBullets.reserve(1500000); // we can have at most ~1 million bullets at once
}

void BulletManager::Update(float time_sec) {

}

void BulletManager::Fire(const DirectX::XMFLOAT3& pos, const DirectX::XMFLOAT3& dir, float speed, float time, float life_time) {
    mBullets.emplace_back(XMLoadFloat3(&pos), XMLoadFloat3(&dir), speed, time, life_time);
}
