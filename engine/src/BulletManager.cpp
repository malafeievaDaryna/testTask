#include "BulletManager.h"
#include <limits>

using namespace DirectX;

namespace {
static float EPSILON = std::numeric_limits<float>::epsilon();
}

BulletManager::BulletManager(std::vector<utils::Wall>& walls) : mWalls(walls) {
    mBullets.reserve(1500000);  // we can have at most ~1 million bullets at once
}

// return false if there is no intersection, the time and point of intersection returned over param out_...
bool BulletManager::getTimeOfIntersection(float time_sec, const Bullet& bullet, const utils::Wall& wall,
                                          float& out_timeIntersection,
                                          XMFLOAT3& out_intersectionPoint) {
    // checking for intersection of ray with plane 
    // by placing line\ray equation R(t) = R0 + t * Rd into plane equation  Ax + By + Cz + D = 0
    // => A(X0 + Xd * t) + B(Y0 + Yd * t) + C(Z0 + Zd * t) + D = 0
    XMVECTOR N = wall.normal;                // A B C from plane equation
    XMVECTOR P = bullet.pos;                 // start pos of the ray
    XMVECTOR V = time_sec * bullet.speed * bullet.dir;  // velocity of the line equation or acceleration dir * time
    float D = wall.distanceToOrigin;         // D from plane equation

    float& t = out_timeIntersection;

    // for getting time of intersection
    // t = -(AX0 + BY0 + CZ0 + D) / (AXd + BYd + CZd)
    // t = -(N · P + D) / (N · V) where (N · V) is denom
    float denom = XMVectorGetX(DirectX::XMVector3Dot(N, V));

    // preventing from dividing by zero if the ray is parallel to the plane and there is no intersection:
    if (abs(denom) <= EPSILON) {
        utils::log_debug("no intersection");
        return false;
    }

    // t = -(N · P + D) / denom;
    t = -(XMVectorGetX(XMVector3Dot(N, P)) + D) / denom;

    // If t < 0 then the ray intersects plane behind origin, i.e. no intersection of interest
    if (t <= EPSILON) {
        utils::log_debug("no intersection");
        return false;
    }

    XMStoreFloat3(&out_intersectionPoint, XMVectorAdd(P, t * V));

    // the plain is limited in width and height and we should ensure that point of intersection fots into bounding box of wall
    if (out_intersectionPoint.x >= wall.leftX && out_intersectionPoint.x <= wall.rightX &&
        out_intersectionPoint.y >= wall.bottomY && out_intersectionPoint.y <= wall.topY) {
        return true;
    }

    return false;
}

void BulletManager::Update(float time_sec) {
    utils::log_debug("cur time s", time_sec);
    auto& wall = mWalls[0];
    auto& bullet = mBullets[0];
    float timeIntersection = 0.0f;
    XMFLOAT3 intersectionPoint;
    if (getTimeOfIntersection(time_sec, bullet, wall, timeIntersection, intersectionPoint)) {
        /*utils::log_debug("intersection happens at point", intersectionPoint.x, intersectionPoint.y, intersectionPoint.z,
                            " time ",
                         timeIntersection);*/
        wall.isDestroyed = timeIntersection <= 1.0f;
    }
}

void BulletManager::Fire(const DirectX::XMFLOAT3& pos, const DirectX::XMFLOAT3& dir, float speed, float time, float life_time) {
    mBullets.emplace_back(XMLoadFloat3(&pos), XMLoadFloat3(&dir), speed, time, life_time);
}
