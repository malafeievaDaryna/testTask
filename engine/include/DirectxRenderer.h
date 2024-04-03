#pragma once

#define NOMINMAX

#include <d3d12.h>
#include <dxgi.h>
#include <wrl.h>  // ComPtr template (kinda smart pointers for COM objects)
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <directxmath.h>
#include "Utils.h"
#include "BulletManager.h"

class Window;

class DirectXRenderer {
    static constexpr uint32_t BULLETS_AMOUNT = 1000000;
    static constexpr uint32_t WALLS_AMOUNT = 100000;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;
    static constexpr DXGI_FORMAT DEPTH_FORMAT = DXGI_FORMAT_D32_FLOAT;
    static constexpr float zFar = 200000.0f;

    // WALL DATA BEING STORED IN ABSOLUTE GPU MEMORY (destroying is visualized by triangle discarding to avoid permanent updating)
    struct WallVertex {
        uint32_t extrudingVertexID; // id of Instance matrix row keeping extruding vertex
        float uv[2];
    };

    // Declare upload buffer data as 'static' so it persists after returning from this function.
    // Otherwise, we would need to explicitly wait for the GPU to copy data from the upload buffer
    // to vertex/index default buffers due to how the GPU processes commands asynchronously.
    static inline const WallVertex WallVertices[4] {  // Upper Left
        {0u, {0, 0}},
        // Upper Right
        {1u, {1, 0}},
        // Bottom right
        {2u, {1, 1}},
        // Bottom left
        {3u, {0, 1}}};

    static inline const int WallIndices[6] {0, 1, 2, 2, 3, 0};

    // wall Instance consists of 4 extruding vectors for quad positioning (keeping actual vertices of wall like 4 rows of matrix)
    struct WallInstance {
        DirectX::XMMATRIX extrudingVectors;
    };

    struct ConstantBuffer {
        DirectX::XMMATRIX mvp;
        uint32_t isWallDestroyed[4000]; // we can not store more than 4096 entries due to limitation that's why we pack 32 walls in one uint32
    };
public:
    DirectXRenderer();
    ~DirectXRenderer();

    void Initialize(const std::string& title, int width, int height);
    bool Run();

private:
    void Shutdown();
    void Render();
    void UpdateConstantBuffer();
    void UpdateBulletsBuffer();
    bool CheckTearingSupport();
    void CreateConstantBuffer();
    void CreateMeshBuffers(ID3D12GraphicsCommandList* uploadCommandList);
    void CreateRootSignature();
    void CreatePipelineStateObject();
    void CreateDeviceAndSwapChain();
    void SetupSwapChain();
    void SetupRenderTargets();
    void BulletsSpawnerJob();

private:
    float mGlobalLifeCycleTimeMS = 0.0f;
    float mDeltaTimeMS = 0.0f;
    std::unique_ptr<Window, void (*)(Window*)> mWindow;
    DirectX::XMMATRIX mModel{};
    DirectX::XMMATRIX mView{};
    DirectX::XMMATRIX mProj{};

    D3D12_VIEWPORT mViewport{};
    D3D12_RECT mRectScissor{};
    Microsoft::WRL::ComPtr<IDXGISwapChain> mSwapChain{};
    Microsoft::WRL::ComPtr<ID3D12Device> mDevice{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mRenderTargets[MAX_FRAMES_IN_FLIGHT]{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mDepthStencilBuffer;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> mCommandQueue{};

    HANDLE mFrameFenceEvents[MAX_FRAMES_IN_FLIGHT]{nullptr};
    Microsoft::WRL::ComPtr<ID3D12Fence> mFrameFences[MAX_FRAMES_IN_FLIGHT]{};
    UINT64 mCurrentFenceValue{0u};
    UINT64 mFenceValues[MAX_FRAMES_IN_FLIGHT]{};
    UINT32 m_currentFrame{0u};

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mDSDescriptorHeap; // the heap for Depth Stencil buffer descriptor
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mRenderTargetDescriptorHeap{};
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mImGuiSrvDescHeap{};
    UINT64 mRenderTargetViewDescriptorSize{0u};

    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature{};
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPsoWall{};
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPsoBullet{};

    Microsoft::WRL::ComPtr<ID3D12Resource> mWallUploadBuffer{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mWallVertexBuffer{};
    D3D12_VERTEX_BUFFER_VIEW mWallVertexBufferView;

    Microsoft::WRL::ComPtr<ID3D12Resource> mWallInstanceBuffer;
    D3D12_VERTEX_BUFFER_VIEW mWallInstanceBufferView;

    Microsoft::WRL::ComPtr<ID3D12Resource> mWallIndexBuffer{};
    D3D12_INDEX_BUFFER_VIEW mWallIndexBufferView{};
    ConstantBuffer mConstantBufferData{};
    Microsoft::WRL::ComPtr<ID3D12Resource> mConstantBuffers[MAX_FRAMES_IN_FLIGHT];
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> mCommandAllocators[MAX_FRAMES_IN_FLIGHT]{};
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> mCommandLists[MAX_FRAMES_IN_FLIGHT]{};

    Microsoft::WRL::ComPtr<ID3D12Resource> mBulletsBuffer{}; // BULLETS BUFFER BEING STORED ON GPU\CPU accessible memory since it's constantly changing
    D3D12_VERTEX_BUFFER_VIEW mBulletsBufferView{};
    utils::Texture2DResource mBulletTextureRes{};

    utils::Texture2DResource mWallTextureRes{};
    std::vector<utils::Wall> mWalls{};
    uint32_t mActualWallsAmount{0u}; // all walls including destroyed ones persist for a while but will be discarded and marked as unused
    std::vector<WallInstance> mWallInstances{};  // WALL DATA BEING STORED IN ABSOLUTE GPU MEMORY (destroying is visualized by triangle discarding to avoid permanent updating)
    BulletManager mBulletMngr;
    std::thread mBulletsSpawnerThread{};
    std::atomic_flag mBulletsSpawnerThreadInterupter{ATOMIC_FLAG_INIT};
    std::atomic_flag mBulletsSpawningStops{ATOMIC_FLAG_INIT};
};
