#include "BulletManager.h"
#include <limits>

using namespace DirectX;

namespace {
static float EPSILON = std::numeric_limits<float>::epsilon();
}

BulletManager::BulletManager(std::vector<utils::Wall>& walls) : mWalls(walls) {
    mBullets.reserve(1000000);  // we can have at most ~1 million bullets at once
    mBulletsSwap.reserve(1000000);
    mBulletInstances.reserve(1000000);
}

// return false if there is no intersection, the time and point of intersection returned over param out_...
bool BulletManager::getTimeOfIntersection(float time_sec, const Bullet& bullet, const utils::Wall& wall,
                                          float& out_timeIntersection, XMFLOAT3& out_intersectionPoint) {
    // checking for intersection of ray with plane
    // by placing line\ray equation R(t) = R0 + t * Rd into plane equation  Ax + By + Cz + D = 0
    // => A(X0 + Xd * t) + B(Y0 + Yd * t) + C(Z0 + Zd * t) + D = 0
    XMVECTOR N = wall.normal;                           // A B C from plane equation
    XMVECTOR P = bullet.pos;                            // start pos of the ray
    XMVECTOR V = time_sec * bullet.speed * bullet.dir;  // velocity of the line equation or acceleration dir * time
    float D = wall.distanceToOrigin;                    // D from plane equation

    float& t = out_timeIntersection;

    // for getting time of intersection
    // t = -(AX0 + BY0 + CZ0 + D) / (AXd + BYd + CZd)
    // t = -(N · P + D) / (N · V) where (N · V) is denom
    float denom = XMVectorGetX(DirectX::XMVector3Dot(N, V));

    // preventing from dividing by zero if the ray is parallel to the plane and there is no intersection:
    if (abs(denom) <= EPSILON) {
        // utils::log_debug("no intersection");
        return false;
    }

    // t = -(N · P + D) / denom;
    t = -(XMVectorGetX(XMVector3Dot(N, P)) + D) / denom;

    // If t < 0 then the ray intersects plane behind origin, i.e. no intersection of interest
    if (t <= EPSILON) {
        // utils::log_debug("no intersection");
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
    std::lock_guard<std::mutex> lock(mSyncMngr);
    float timeOfCollisionWithWallS = 0.0f;
    XMFLOAT3 intersectionPoint;
    mBulletsSwap.clear();
    mBulletInstances.clear();
    for (auto& bullet : mBullets) {
        bool checkingWallsNeeded = true;
        // if we already know about the posible collision then we don't need to make any calculations twice
        if (bullet.isPrecalculationCollision) {
            // if the wall being checked is already destroyed then look for new wall from the beginning
            if (bullet.idOfTheWall >= 0 && mWalls[bullet.idOfTheWall].isDestroyed) {
                bullet.isPrecalculationCollision = false;
                bullet.idOfTheWall = -1;
            } else if (bullet.idOfTheWall >= 0 && (time_sec - bullet.time_creation) >= bullet.timeOfCollisionWithWallS) {
                mWalls[bullet.idOfTheWall].isDestroyed = true;
                // utils::log_debug("WALL destroyed");
                // TODO reflection;
                continue;
            } else {
                // no any collisions with this bullet
                // no need to make any calculations twice let's skip all walls for this bullet
                checkingWallsNeeded = false;
            }
        }
        if (checkingWallsNeeded) {
            for (size_t j = 0u; j < mWalls.size(); ++j) {
                auto& wall = mWalls[j];
                if (wall.isDestroyed) {
                    // the wall will be eliminated at next pass
                    continue;
                }
                // since our walls are ordered by Z value in ascending way then we can skip walls which are too far from final
                // position of the bullet
                if ((bullet.distanceToOrigin + EPSILON) < wall.distanceToOrigin) {
                    break;
                }
                if (getTimeOfIntersection(time_sec - bullet.time_creation, bullet, wall, timeOfCollisionWithWallS,
                                          intersectionPoint)) {
                    /*utils::log_debug("intersection happens at point", intersectionPoint.x, intersectionPoint.y,
                                        intersectionPoint.z, " time ", timeIntersection);*/
                    // now we know which wall the bullet is going to intersect
                    bullet.isPrecalculationCollision = true;
                    bullet.idOfTheWall = j;
                    bullet.timeOfCollisionWithWallS = timeOfCollisionWithWallS;
                    break;
                }
            }
            // we know that the bullet doesn't intersects any wall (bullet.idOfTheWall == -1)
            bullet.isPrecalculationCollision = true;
        }

        // checking whether the bullet is dead
        if (bullet.time_creation + bullet.life_time < time_sec) {
            // ensuring that the wall is destroyed for sure since the bullet is dead
            if (bullet.isPrecalculationCollision && bullet.idOfTheWall >= 0) {
                mWalls[bullet.idOfTheWall].isDestroyed = true;
            }
        } else {
            auto posVec = XMVectorAdd(bullet.pos, (time_sec - bullet.time_creation) * bullet.speed * bullet.dir);
            XMFLOAT3 pos;
            XMStoreFloat3(&pos, posVec);
            mBulletInstances.emplace_back(pos);
            mBulletsSwap.push_back(bullet);  // stores in swap buffer
        }
    }

    mBullets.clear();
    if (!mBulletsSwap.empty()) {
        mBullets = mBulletsSwap;
    }
}

void BulletManager::Fire(const DirectX::XMVECTOR& pos, const DirectX::XMVECTOR& dir, float speed, float time, float life_time) {
    std::lock_guard<std::mutex> lock(mSyncMngr);
    float distanceToOrigin =
        XMVectorGetX(XMVector3Length(XMVectorAdd(pos, life_time * speed * dir)));  // length of the final position of the bullet
    mBullets.emplace_back(pos, dir, speed, time, life_time, distanceToOrigin);
}
