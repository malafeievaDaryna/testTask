#include "DirectXRenderer.h"
#include "Window.h"
/**
 * d3dx12.h provides some useful classes that will simplify some of the functions
 * it needs to be downloaded separately from the Microsoft DirectX repository
 * (https://github.com/Microsoft/DirectX-Graphics-Samples/tree/master/Libraries/D3DX12)
 */
#include "Shaders.h"
#include "d3dx12.h"

#include <d3dcompiler.h>
#include <dxgi1_5.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <random>
#include <map>

#include <imgui_impl_win32.h>
#include <imgui_impl_dx12.h>
#include <imgui.h>

// for debugging on poor GPU uncomment the following line
// #define USING_INTEGRATED_GPU

// ALOW_TEARING for G-SYNC monitor
// #define ALOW_TEARING

using namespace Microsoft::WRL;
using namespace DirectX;
using namespace utils;

namespace {
// GPU/ CPU accessive memory
static const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
static const XMVECTOR EYE = XMVectorSet(0, 200, -700, 1);
static const XMVECTOR TARGET = XMVectorSet(0, 0, 0, 1);
static const XMVECTOR UP = XMVectorSet(0, 1, 0, 0);
struct RenderEnvironment {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain> swapChain;
};
void WaitForFence(ID3D12Fence* fence, UINT64 completionValue, HANDLE waitEvent) {
    if (fence->GetCompletedValue() < completionValue) {
        fence->SetEventOnCompletion(completionValue, waitEvent);
        WaitForSingleObject(waitEvent, INFINITE);
    }
}

RenderEnvironment CreateDeviceAndSwapChainHelper(D3D_FEATURE_LEVEL minimumFeatureLevel, const DXGI_SWAP_CHAIN_DESC* swapChainDesc) {
    RenderEnvironment result;

    ComPtr<IDXGIFactory4> dxgiFactory;
    auto hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));
    ComPtr<IDXGIAdapter1> dxgiAdapter;
    if (FAILED(hr)) {
        throw std::runtime_error("DXGI factory creation failed.");
    }

    SIZE_T maxDedicatedVideoMemory = 0;
    UINT gpuAdapterId = 0u;
    DXGI_ADAPTER_DESC1 dxgiAdapterDesc;
    for (UINT i = 0u; dxgiFactory->EnumAdapters1(i, &dxgiAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        dxgiAdapter->GetDesc1(&dxgiAdapterDesc);

        // let's try to pickup the discrete gpu (filtering by dedicated video memory that gpu will be favored)
        if ((dxgiAdapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
            SUCCEEDED(D3D12CreateDevice(dxgiAdapter.Get(), minimumFeatureLevel, __uuidof(ID3D12Device), nullptr)) &&
            dxgiAdapterDesc.DedicatedVideoMemory > maxDedicatedVideoMemory) {
            gpuAdapterId = i;
            maxDedicatedVideoMemory = dxgiAdapterDesc.DedicatedVideoMemory;
        }
    }
#if defined(_DEBUG) && defined(USING_INTEGRATED_GPU)
    // enforce using integrated gpu if it exists for testing on poor hardware
    if (dxgiFactory->EnumAdapters1(1u, &dxgiAdapter) != DXGI_ERROR_NOT_FOUND) {
        gpuAdapterId = 1u;
        dxgiAdapter->GetDesc1(&dxgiAdapterDesc);
    }
#endif
     
    dxgiFactory->EnumAdapters1(gpuAdapterId, &dxgiAdapter);
    ComPtr<IDXGIAdapter> adapter;
    dxgiAdapter.As(&adapter);
    hr = D3D12CreateDevice(adapter.Get(), minimumFeatureLevel, IID_PPV_ARGS(&result.device));

    if (FAILED(hr)) {
        throw std::runtime_error("Device creation failed.");
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    hr = result.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&result.queue));

    if (FAILED(hr)) {
        throw std::runtime_error("Command queue creation failed.");
    }

    // Must copy into non-const space
    DXGI_SWAP_CHAIN_DESC swapChainDescCopy = *swapChainDesc;
    hr = dxgiFactory->CreateSwapChain(result.queue.Get(), &swapChainDescCopy, &result.swapChain);

    if (FAILED(hr)) {
        throw std::runtime_error("Swap chain creation failed.");
    }

    return result;
}
}  // namespace

DirectXRenderer::DirectXRenderer()
    : mWindow{nullptr, nullptr}, mWallInstances(WALLS_AMOUNT, WallInstance{}), mBulletMngr(mWalls) {
    mWalls.reserve(100000); // we can have at most ~100000 walls at once
}

DirectXRenderer::~DirectXRenderer() {
    Shutdown();
}

void DirectXRenderer::Render() {
    static auto startTime = std::chrono::high_resolution_clock::now();
    static auto endTime = std::chrono::high_resolution_clock::now();

    //FPS UI
    static uint32_t framesCounter = 0u;
    static float secondElapsed = 0.0f; // when second elapsed then we measure fps
    static uint32_t FPS = 0u;
    static uint32_t MIN_FPS = std::numeric_limits<uint32_t>::max();
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("Menu", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);
    ImGui::SetWindowPos(ImVec2(0, 0));
    ImGui::SetWindowSize(ImVec2(0, 0));
    ImGui::SetWindowFontScale(1.3);
    ImGui::Text("FPS %d", FPS);
    ImGui::Text("MIN FPS %d", MIN_FPS);
    ImGui::Text("WALLS AMOUNT %d", mActualWallsAmount);
    ImGui::Text("BULLETS AMOUNT %d", mBulletMngr.getBulletsAmount());
    if (ImGui::Button("spawn 1000000 bullets", ImVec2(200, 50))) {
        mBulletsSpawningStops.clear(std::memory_order_relaxed);
    }
    ImGui::End();
    ImGui::Render();

    // waiting for completion of frame processing on gpu
    WaitForFence(mFrameFences[m_currentFrame].Get(), mFenceValues[m_currentFrame], mFrameFenceEvents[m_currentFrame]);

    mCommandAllocators[m_currentFrame]->Reset();

    auto commandList = mCommandLists[m_currentFrame].Get();
    commandList->Reset(mCommandAllocators[m_currentFrame].Get(), nullptr);

    // prepare RenderTargets\Depth handlers
    D3D12_CPU_DESCRIPTOR_HANDLE renderTargetHandle;
    CD3DX12_CPU_DESCRIPTOR_HANDLE::InitOffsetted(renderTargetHandle,
                                                 mRenderTargetDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
                                                 m_currentFrame, mRenderTargetViewDescriptorSize);
    CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(mDSDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

    commandList->OMSetRenderTargets(1, &renderTargetHandle, true, &dsvHandle);
    commandList->RSSetViewports(1, &mViewport);
    commandList->RSSetScissorRects(1, &mRectScissor);

    D3D12_RESOURCE_BARRIER barrierBefore;
    barrierBefore.Transition.pResource = mRenderTargets[m_currentFrame].Get();
    barrierBefore.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrierBefore.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrierBefore.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrierBefore.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrierBefore.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    commandList->ResourceBarrier(1, &barrierBefore);

    mBulletMngr.Update(mGlobalLifeCycleTimeMS / 1000.0f);
    UpdateBulletsBuffer();
    UpdateConstantBuffer();

    static const float clearColor[] = {1.0f, 1.0f, 1.0f, 1.0f};

    commandList->ClearRenderTargetView(renderTargetHandle, clearColor, 0, nullptr);
    commandList->ClearDepthStencilView(mDSDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0,
                                       0, nullptr);
    // WALLS
    commandList->SetPipelineState(mPsoWall.Get());
    commandList->SetGraphicsRootSignature(mRootSignature.Get());
    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    // Set the descriptor heap containing the texture srv
    ID3D12DescriptorHeap* heaps[] = {mWallTextureRes.srvDescriptorHeap.Get()};
    commandList->SetDescriptorHeaps(1, heaps);
    // Set slot 0 of our root signature to point to our descriptor heap with
    // the texture SRV
    commandList->SetGraphicsRootDescriptorTable(0, mWallTextureRes.srvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
    // Set slot 1 of our root signature to the constant buffer view
    commandList->SetGraphicsRootConstantBufferView(1, mConstantBuffers[m_currentFrame]->GetGPUVirtualAddress());

    commandList->IASetVertexBuffers(0, 1, &mWallVertexBufferView);
    commandList->IASetVertexBuffers(1, 1, &mWallInstanceBufferView);
    commandList->IASetIndexBuffer(&mWallIndexBufferView);
    commandList->DrawIndexedInstanced(6, mWalls.size(), 0, 0, 0); // drawing 100000 instances of the wall

    // BULLETS
    commandList->SetPipelineState(mPsoBullet.Get());
    // Set the descriptor heap containing the texture srv
    ID3D12DescriptorHeap* bullet_heaps[] = {mBulletTextureRes.srvDescriptorHeap.Get()};
    commandList->SetDescriptorHeaps(1, bullet_heaps);
    // Set slot 0 of our root signature to point to our descriptor heap with
    // the texture SRV
    commandList->SetGraphicsRootDescriptorTable(0, mBulletTextureRes.srvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
    // Set slot 1 of our root signature to the constant buffer view
    commandList->SetGraphicsRootConstantBufferView(1, mConstantBuffers[m_currentFrame]->GetGPUVirtualAddress());
    commandList->IASetVertexBuffers(0, 1, &mBulletsBufferView);
    commandList->DrawInstanced(6, mBulletMngr.getBulletsAmount(), 0, 0);  // drawing instances of the bullet

    // IMGUI UI
    ID3D12DescriptorHeap* imgui_heaps[] = {mImGuiSrvDescHeap.Get()};
    commandList->SetDescriptorHeaps(1, imgui_heaps);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), commandList);

    D3D12_RESOURCE_BARRIER barrierAfter;
    barrierAfter.Transition.pResource = mRenderTargets[m_currentFrame].Get();
    barrierAfter.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrierAfter.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrierAfter.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrierAfter.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barrierAfter.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    commandList->ResourceBarrier(1, &barrierAfter);

    commandList->Close();

    ID3D12CommandList* commandLists[] = {commandList};
    mCommandQueue->ExecuteCommandLists(std::extent<decltype(commandLists)>::value, commandLists);

    mSwapChain->Present(CheckTearingSupport() ? 0 : 1, 0);

    // the value the gpu will set when preseting finished
    const auto fenceValue = mCurrentFenceValue;
    mCommandQueue->Signal(mFrameFences[m_currentFrame].Get(), fenceValue);
    mFenceValues[m_currentFrame] = fenceValue;
    ++mCurrentFenceValue;

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    assert(m_currentFrame < MAX_FRAMES_IN_FLIGHT);

    endTime = std::chrono::high_resolution_clock::now();
    mDeltaTimeMS = std::chrono::duration<float, std::chrono::milliseconds::period>(endTime - startTime).count();
    mGlobalLifeCycleTimeMS += mDeltaTimeMS;
    startTime = endTime;

    // FPS measuring
    secondElapsed += mDeltaTimeMS;
    framesCounter++;
    if (secondElapsed >= 1000.0) { // how many frames were drawn per second
        FPS = framesCounter;
        MIN_FPS = MIN_FPS > FPS ? FPS : MIN_FPS;
        framesCounter = 0u;
        secondElapsed = 0.0;
    }
}

bool DirectXRenderer::Run() {
    MSG msg;
    // regular events loop to get window responsive
    // checking msg in the window queue
    if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        /* handle or dispatch messages */
        if (msg.message == WM_QUIT) {
            return false;
        } else {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    Render();

    return true;
}

/**
Setup all render targets. This creates the render target descriptor heap and
render target views for all render targets.
This function does not use a default view but instead changes the format to
_SRGB.
*/
void DirectXRenderer::SetupRenderTargets() {
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = MAX_FRAMES_IN_FLIGHT;
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(mDevice->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&mRenderTargetDescriptorHeap)))) {
        log_err("Couldn't allocate gpu heap memory");
    }

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle{mRenderTargetDescriptorHeap->GetCPUDescriptorHandleForHeapStart()};

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        D3D12_RENDER_TARGET_VIEW_DESC viewDesc;
        viewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        viewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        viewDesc.Texture2D.MipSlice = 0;
        viewDesc.Texture2D.PlaneSlice = 0;

        mDevice->CreateRenderTargetView(mRenderTargets[i].Get(), &viewDesc, rtvHandle);

        rtvHandle.Offset(mRenderTargetViewDescriptorSize);
    }

    // create a depth stencil descriptor heap
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
    dsvHeapDesc.NumDescriptors = 1;
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(mDevice->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&mDSDescriptorHeap)))) {
        log_err("Couldn't allocate gpu heap memory");
    }

    D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
    depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;

    D3D12_CLEAR_VALUE depthOptimizedClearValue = {};
    depthOptimizedClearValue.Format = DEPTH_FORMAT;
    depthOptimizedClearValue.DepthStencil.Depth = 1.0f;
    depthOptimizedClearValue.DepthStencil.Stencil = 0;

    mDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
                                     &CD3DX12_RESOURCE_DESC::Tex2D(DEPTH_FORMAT, mWindow->width(), mWindow->height(), 1, 0, 1, 0,
                                                                   D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                                     D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthOptimizedClearValue,
                                     IID_PPV_ARGS(&mDepthStencilBuffer));
    mDSDescriptorHeap->SetName(L"Depth/Stencil Resource Heap");

    mDevice->CreateDepthStencilView(mDepthStencilBuffer.Get(), &depthStencilDesc,
                                    mDSDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
}

/**
Set up swap chain related resources, that is, the render target view, the
fences, and the descriptor heap for the render target.
*/
void DirectXRenderer::SetupSwapChain() {
    mCurrentFenceValue = 1;

    // Create fences for each frame so we can protect resources and wait for
    // any given frame
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        mFrameFenceEvents[i] = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        mFenceValues[i] = 0;
        mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mFrameFences[i]));
    }

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        mSwapChain->GetBuffer(i, IID_PPV_ARGS(&mRenderTargets[i]));
    }

    SetupRenderTargets();
}

void DirectXRenderer::BulletsSpawnerJob() {
    using namespace std::chrono_literals;
    //-----------------------------------------------------//
    // THE BULLET GENERATOR
    std::random_device rd;
    std::mt19937 gen(rd());  // seed the generator
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
    std::default_random_engine generator;
    static const XMVECTOR minRotation = XMQuaternionRotationNormal(UP, XMConvertToRadians(-60.0f));
    static const XMVECTOR maxRotation = XMQuaternionRotationNormal(UP, XMConvertToRadians(60.0f));
    static const XMVECTOR bulletStartPos = XMVectorSet(0.0f, 0.0f, -500.0f, 1.0f);
    static const XMVECTOR defaultBulletDir = XMVectorSet(0.0f, 0.0f, 1.0f, 1.0f);  // the default direction is along the Z axis

    mBulletsSpawningStops.test_and_set(std::memory_order_relaxed); // initiate with true value to suppress creation of bullets
    while (mBulletsSpawnerThreadInterupter.test_and_set(std::memory_order_relaxed)) {
        if (mBulletsSpawningStops.test_and_set(std::memory_order_relaxed)) {
            // bullets generator inactive
            continue;
        }
        for (std::size_t i = 0u; i < BULLETS_AMOUNT; ++i) {       // 1 million bullets
            float factorInterpolation = randomFloats(generator);  // some percentage between quaternions
            auto interpolatedQuatRotation =
                XMQuaternionSlerp(minRotation, maxRotation,
                                  factorInterpolation);  // getting random interpolated rotation between two min\max quaternions
            // multiplying default dir by interpolated rotation quat provides us random rotation
            XMVECTOR bulletDir = XMQuaternionMultiply(XMQuaternionMultiply(interpolatedQuatRotation, defaultBulletDir),
                                                      XMQuaternionConjugate(interpolatedQuatRotation));
#if defined(_DEBUG)
            XMFLOAT3 _bulletDir;
            XMStoreFloat3(&_bulletDir, bulletDir);
            // log_debug("random bullet dir:", _bulletDir.x, _bulletDir.y, _bulletDir.z);
#endif
            float speed = 100.0f + randomFloats(generator) * 500.0f;  // m/s
            float timeOfCreationS = mGlobalLifeCycleTimeMS / 1000.0f;
            float lifeTime = randomFloats(generator) * 60.0f; // 1min
            mBulletMngr.Fire(bulletStartPos, bulletDir, speed, timeOfCreationS, lifeTime);
            std::this_thread::sleep_for(1ms);
            if (!mBulletsSpawnerThreadInterupter.test_and_set(std::memory_order_relaxed)) {
                return;
            }
        }
    }
}

void DirectXRenderer::Initialize(const std::string& title, int width, int height) {
    // Check for DirectX Math library support.
    if (!XMVerifyCPUSupport()) {
        MessageBoxA(NULL, "Failed to verify DirectX Math library support.", "Error", MB_OK | MB_ICONERROR);
        std::exit(-1);
    }

    mWindow.reset(new Window("DirectXMiniGame", 1280, 720));
    mWindow.get_deleter() = [](Window* ptr) { delete ptr; };

    mView = XMMatrixLookAtLH(EYE, TARGET, UP);
    float aspectRatio = static_cast<float>(mWindow->width()) / mWindow->height();
    mProj = XMMatrixPerspectiveFovLH(XMConvertToRadians(45.0f), aspectRatio, 0.01f, zFar);

    CreateDeviceAndSwapChain();

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.WantCaptureMouse = true;
    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init(mWindow->hwnd());
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = 1;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (mDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&mImGuiSrvDescHeap)) != S_OK)
            log_err("CreateDescriptorHeap failed for imgui");
    }
    ImGui_ImplDX12_Init(mDevice.Get(), MAX_FRAMES_IN_FLIGHT, DXGI_FORMAT_R8G8B8A8_UNORM, mImGuiSrvDescHeap.Get(),
                        mImGuiSrvDescHeap->GetCPUDescriptorHandleForHeapStart(),
                        mImGuiSrvDescHeap->GetGPUDescriptorHandleForHeapStart());

    mRectScissor = {0, 0, (long)mWindow->width(), (long)mWindow->height()};
    mViewport = {0.0f, 0.0f, static_cast<float>(mWindow->width()), static_cast<float>(mWindow->height()), 0.0f, 1.0f};

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        mDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&mCommandAllocators[i]));
        mDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, mCommandAllocators[i].Get(), nullptr,
                                   IID_PPV_ARGS(&mCommandLists[i]));
        mCommandLists[i]->Close();
    }
    // Create our upload fence, command list and command allocator
    // This will be only used while creating the mesh buffer and the texture
    // to upload data to the GPU.
    ComPtr<ID3D12Fence> uploadFence;
    mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&uploadFence));

    ComPtr<ID3D12CommandAllocator> uploadCommandAllocator;
    mDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&uploadCommandAllocator));
    ComPtr<ID3D12GraphicsCommandList> uploadCommandList;
    mDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, uploadCommandAllocator.Get(), nullptr,
                               IID_PPV_ARGS(&uploadCommandList));
    CreateRootSignature();
    CreatePipelineStateObject();
    CreateMeshBuffers(uploadCommandList.Get());
    CreateConstantBuffer();
    mWallTextureRes = CreateTexture(mDevice.Get(), uploadCommandList.Get(), "wall.jpg");
    mBulletTextureRes = CreateTexture(mDevice.Get(), uploadCommandList.Get(), "bullet.png");

    uploadCommandList->Close();

    // Execute the upload and finish the command list
    ID3D12CommandList* commandLists[] = {uploadCommandList.Get()};
    mCommandQueue->ExecuteCommandLists(std::extent<decltype(commandLists)>::value, commandLists);
    mCommandQueue->Signal(uploadFence.Get(), 1);

    auto waitEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    if (waitEvent == NULL) {
        throw std::runtime_error("Could not create wait event.");
    }

    WaitForFence(uploadFence.Get(), 1, waitEvent);

    // Cleanup our upload handle
    uploadCommandAllocator->Reset();

    CloseHandle(waitEvent);
}

void DirectXRenderer::CreateMeshBuffers(ID3D12GraphicsCommandList* uploadCommandList) {
    //-----------------------------------------------------//
    // THE WALL GENERATOR
    std::random_device rd;
    std::mt19937 gen(rd());  // seed the generator
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
    std::default_random_engine generator;
    static const XMVECTOR NORMAL = XMVectorSet(0.0f, 0.0f, -1.0f, 1.0f);
    static const XMVECTOR SIDE = XMVector3Cross(NORMAL, UP);
    utils::Wall wall;
    std::multimap<float, utils::Wall> sortedWallsByZ;
    for (std::size_t i = 0u; i < WALLS_AMOUNT; ++i) {
        float centerZ = randomFloats(generator) * WALLS_AMOUNT;  // in positive Z axis
        float centerX = (2.0f * randomFloats(generator) - 1.0f) * 0.3 * 2000;
        float centerY = 0.0f;
        wall.center = XMVectorSet(centerX, centerY, centerZ, 1.0f);
        wall.distanceToOrigin = XMVectorGetX(XMVector3Length(wall.center));
        wall.width = randomFloats(generator) * 100.0f;
        wall.height = randomFloats(generator) * 100.0f;
        wall.normal = NORMAL;
        XMFLOAT3 top, bottom, left, right;
        XMStoreFloat3(&top, XMVectorAdd(wall.center, 0.5f * wall.height * UP));
        XMStoreFloat3(&bottom, XMVectorAdd(wall.center, -0.5f * wall.height * UP));
        XMStoreFloat3(&left, XMVectorAdd(wall.center, -0.5f * wall.width * SIDE));
        XMStoreFloat3(&right, XMVectorAdd(wall.center, 0.5f * wall.width * SIDE));
        wall.topY = top.y;
        wall.bottomY = bottom.y;
        wall.leftX = left.x;
        wall.rightX = right.x;
        wall.centerZ = centerZ;
        sortedWallsByZ.emplace(centerZ, wall);
    }

    // Important !!! sorting walls by ascending order of Z component to get benefit of early depth test
    std::size_t i = 0u;
    float zBias = 0.01f;
    for (auto& item : sortedWallsByZ) {
        auto& wall = item.second;
        //wall.centerZ += zBias;
        mWalls.push_back(wall);
        mWallInstances[i].extrudingVectors.r[0] = XMVectorSet(wall.leftX, wall.topY, wall.centerZ, 1.0f);  // Upper Left
        mWallInstances[i].extrudingVectors.r[1] = XMVectorSet(wall.rightX, wall.topY, wall.centerZ, 1.0f);  // Upper Right
        mWallInstances[i].extrudingVectors.r[2] = XMVectorSet(wall.rightX, wall.bottomY, wall.centerZ, 1.0f);  // Bottom right
        mWallInstances[i].extrudingVectors.r[3] = XMVectorSet(wall.leftX, wall.bottomY, wall.centerZ, 1.0f);   // Bottom left
        // since picking rows in vertex shader returns me collumns we need to transpose matrix
        mWallInstances[i].extrudingVectors = XMMatrixTranspose(mWallInstances[i].extrudingVectors);
        ++i;
        zBias += 0.01f;
    }
    mActualWallsAmount = mWalls.size();

    static const int wallsBytesCapacity = mWallInstances.size() * sizeof(WallInstance);
    static const int uploadBufferSize = sizeof(WallVertices) + sizeof(WallIndices) + wallsBytesCapacity;
    static const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    static const auto uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize);

    // Create upload buffer on CPU
    mDevice->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mWallUploadBuffer));

    // Create vertex & index buffer on the GPU
    // HEAP_TYPE_DEFAULT is on GPU, we also initialize with COPY_DEST state
    // so we don't have to transition into this before copying into them
    static const auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    static const auto vertexBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(WallVertices));
    mDevice->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &vertexBufferDesc,
                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&mWallVertexBuffer));

    static const auto indexBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(WallIndices));
    mDevice->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &indexBufferDesc,
                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&mWallIndexBuffer));

    static const auto instanceBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(wallsBytesCapacity);
    mDevice->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &instanceBufferDesc,
                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&mWallInstanceBuffer));

    // Create buffer views
    mWallVertexBufferView.BufferLocation = mWallVertexBuffer->GetGPUVirtualAddress();
    mWallVertexBufferView.SizeInBytes = sizeof(WallVertices);
    mWallVertexBufferView.StrideInBytes = sizeof(WallVertex);

    mWallIndexBufferView.BufferLocation = mWallIndexBuffer->GetGPUVirtualAddress();
    mWallIndexBufferView.SizeInBytes = sizeof(WallIndices);
    mWallIndexBufferView.Format = DXGI_FORMAT_R32_UINT;

    // Create buffer views
    mWallInstanceBufferView.BufferLocation = mWallInstanceBuffer->GetGPUVirtualAddress();
    mWallInstanceBufferView.SizeInBytes = wallsBytesCapacity;
    mWallInstanceBufferView.StrideInBytes = sizeof(WallInstance);

    // Copy data on CPU into the upload buffer
    void* p;
    mWallUploadBuffer->Map(0, nullptr, &p);
    ::memcpy(p, WallVertices, sizeof(WallVertices));
    ::memcpy(static_cast<unsigned char*>(p) + sizeof(WallVertices), WallIndices, sizeof(WallIndices));
    ::memcpy(static_cast<unsigned char*>(p) + sizeof(WallVertices) + sizeof(WallIndices), &mWallInstances[0], wallsBytesCapacity);
    mWallUploadBuffer->Unmap(0, nullptr);

    // Copy data from upload buffer on CPU into the index/vertex buffer on
    // the GPU
    uploadCommandList->CopyBufferRegion(mWallVertexBuffer.Get(), 0, mWallUploadBuffer.Get(), 0, sizeof(WallVertices));
    uploadCommandList->CopyBufferRegion(mWallIndexBuffer.Get(), 0, mWallUploadBuffer.Get(), sizeof(WallVertices),
                                        sizeof(WallIndices));
    uploadCommandList->CopyBufferRegion(mWallInstanceBuffer.Get(), 0, mWallUploadBuffer.Get(),
                                        sizeof(WallVertices) + sizeof(WallIndices), wallsBytesCapacity);

    // Barriers, batch them together
    const CD3DX12_RESOURCE_BARRIER barriers[3] = {
        CD3DX12_RESOURCE_BARRIER::Transition(mWallVertexBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                             D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER),
        CD3DX12_RESOURCE_BARRIER::Transition(mWallIndexBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                             D3D12_RESOURCE_STATE_INDEX_BUFFER),
        CD3DX12_RESOURCE_BARRIER::Transition(mWallInstanceBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                             D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)};

    uploadCommandList->ResourceBarrier(3, barriers);

    // BULLETS BUFFER BEING STORED ON GPU\CPU accessible memory since it's constantly changing
    static const int bulletsBytesCapacity = BULLETS_AMOUNT * sizeof(BulletManager::BulletInstance);
    static const auto uploadBulletsBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bulletsBytesCapacity);

    mDevice->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBulletsBufferDesc,
                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mBulletsBuffer));

    mBulletsBufferView.BufferLocation = mBulletsBuffer->GetGPUVirtualAddress();
    mBulletsBufferView.SizeInBytes = bulletsBytesCapacity;
    mBulletsBufferView.StrideInBytes = sizeof(BulletManager::BulletInstance);

    // now when walls are ready
    mBulletsSpawnerThreadInterupter.test_and_set(std::memory_order_relaxed);
    mBulletsSpawnerThread = std::thread(&DirectXRenderer::BulletsSpawnerJob, this);
}

void DirectXRenderer::CreatePipelineStateObject() {
    // WALL
    static const D3D12_INPUT_ELEMENT_DESC wallLayout[] = {
        {"EXTRUDVERTID", 0, DXGI_FORMAT_R32_UINT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 4, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"INSTANCEEXTRUDINGVERTS", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
        {"INSTANCEEXTRUDINGVERTS", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 16, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
        {"INSTANCEEXTRUDINGVERTS", 2, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 32, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
        {"INSTANCEEXTRUDINGVERTS", 3, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 48, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1}};

#if defined(_DEBUG)
    // Enable better shader debugging with the graphics debugging tools.
    UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    UINT compileFlags = 0;
#endif

    ComPtr<ID3DBlob> errors;
    ComPtr<ID3DBlob> vertexShader;
    HRESULT hr = D3DCompile(shaders::vs_wall_shader, sizeof(shaders::vs_wall_shader), "", nullptr, nullptr, "VS_main", "vs_5_1",
                            compileFlags, 0,
               &vertexShader, &errors);
    if (FAILED(hr) && errors) {
        log_err("error when compiling vs shaders", reinterpret_cast<const char*>(errors.Get()->GetBufferPointer()));
    }

    ComPtr<ID3DBlob> pixelShader;
    hr = D3DCompile(shaders::fs_shader, sizeof(shaders::fs_shader), "", nullptr, nullptr, "PS_main", "ps_5_1", compileFlags, 0,
               &pixelShader, &errors);
    if (FAILED(hr) && errors) {
        log_err("error when compiling fs shaders", reinterpret_cast<const char*>(errors.Get()->GetBufferPointer()));
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.VS.BytecodeLength = vertexShader->GetBufferSize();
    psoDesc.VS.pShaderBytecode = vertexShader->GetBufferPointer();
    psoDesc.PS.BytecodeLength = pixelShader->GetBufferSize();
    psoDesc.PS.pShaderBytecode = pixelShader->GetBufferPointer();
    psoDesc.pRootSignature = mRootSignature.Get();
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    psoDesc.DSVFormat = DEPTH_FORMAT;
    psoDesc.InputLayout.NumElements = std::extent<decltype(wallLayout)>::value;
    psoDesc.InputLayout.pInputElementDescs = wallLayout;
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    // psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE; // both faces drawn
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    // Simple alpha blending
    psoDesc.BlendState.RenderTarget[0].BlendEnable = true;
    psoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
    psoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    psoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    psoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    psoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    psoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.SampleDesc.Count = 1;
    psoDesc.DepthStencilState.DepthEnable = false;
    psoDesc.DepthStencilState.StencilEnable = false;
    psoDesc.SampleMask = 0xFFFFFFFF;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    // let's create default depth testing: DepthEnable = TRUE; DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);

    hr = mDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&mPsoWall));
    if (FAILED(hr)) {
        log_err("error when creating pso for wall");
    }

    // PSO for BULLET
    static const D3D12_INPUT_ELEMENT_DESC bulletLayout[] = {
        {"INSTANCEPOS", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1}};

    ComPtr<ID3DBlob> vertexShaderBullet;
    hr = D3DCompile(shaders::vs_bullet_shader, sizeof(shaders::vs_bullet_shader), "", nullptr, nullptr, "VS_main",
                            "vs_5_1", compileFlags, 0, &vertexShaderBullet, &errors);
    if (FAILED(hr) && errors) {
        log_err("error when compiling vs shaders", reinterpret_cast<const char*>(errors.Get()->GetBufferPointer()));
    }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDescBullet = psoDesc;
    psoDescBullet.InputLayout.NumElements = std::extent<decltype(bulletLayout)>::value;
    psoDescBullet.InputLayout.pInputElementDescs = bulletLayout;
    psoDescBullet.VS.BytecodeLength = vertexShaderBullet->GetBufferSize();
    psoDescBullet.VS.pShaderBytecode = vertexShaderBullet->GetBufferPointer();
    hr = mDevice->CreateGraphicsPipelineState(&psoDescBullet, IID_PPV_ARGS(&mPsoBullet));
    if (FAILED(hr)) {
        log_err("error when creating pso for bullet");
    }
}

void DirectXRenderer::CreateRootSignature() {
    // We have two root parameters, one is a pointer to a descriptor heap
    // with a SRV, the second is a constant buffer view
    CD3DX12_ROOT_PARAMETER parameters[2];

    // Create a descriptor table with one entry in our descriptor heap
    CD3DX12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0};
    parameters[0].InitAsDescriptorTable(1, &range);

    // Our constant buffer view
    parameters[1].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_VERTEX);

    // We don't use another descriptor heap for the sampler, instead we use a
    // static sampler
    CD3DX12_STATIC_SAMPLER_DESC samplers[1];
    samplers[0].Init(0, D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT);

    CD3DX12_ROOT_SIGNATURE_DESC descRootSignature;

    // Create the root signature
    descRootSignature.Init(2, parameters, 1, samplers, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    ComPtr<ID3DBlob> rootBlob;
    ComPtr<ID3DBlob> errorBlob;
    D3D12SerializeRootSignature(&descRootSignature, D3D_ROOT_SIGNATURE_VERSION_1, &rootBlob, &errorBlob);

    mDevice->CreateRootSignature(0, rootBlob->GetBufferPointer(), rootBlob->GetBufferSize(), IID_PPV_ARGS(&mRootSignature));
}

void DirectXRenderer::CreateConstantBuffer() {
    mConstantBufferData.mvp = XMMatrixMultiply(mView, mProj);
    uint32_t sizeOfItem = sizeof(mConstantBufferData.isWallDestroyed[0]);
    uint32_t amount = sizeof(mConstantBufferData.isWallDestroyed) / sizeOfItem;
    memset(mConstantBufferData.isWallDestroyed, std::numeric_limits<int32_t>::max(),
           amount * sizeOfItem);  // all walls are invisible
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        static const auto constantBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(ConstantBuffer));
        mDevice->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &constantBufferDesc,
                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mConstantBuffers[i]));

        void* p;
        mConstantBuffers[i]->Map(0, nullptr, &p);
        memcpy(p, &mConstantBufferData, sizeof(mConstantBufferData));
        mConstantBuffers[i]->Unmap(0, nullptr);
    }
}

void DirectXRenderer::UpdateConstantBuffer() {
    // TODO we can consider not updating all the time since updating needes only when walls changes
    // but due to some restriction like we are able to update only one mConstantBufferData (with currently processed index m_currentFrame) out of three
   mActualWallsAmount = 0u;
   for (std::size_t i = 0u; i < mWalls.size(); ++i) {
       uint32_t index = i / 32;  // producing of packed index
       if (mWalls[i].isDestroyed) {
           mConstantBufferData.isWallDestroyed[index] |= ((uint32_t) 1 << (i % 32));
       } else {
           mConstantBufferData.isWallDestroyed[index] &= ~((uint32_t) 1 << (i % 32));
           ++mActualWallsAmount;
       }
   }
   void* data;
   mConstantBuffers[m_currentFrame]->Map(0, nullptr, &data);
   memcpy(data, &mConstantBufferData, sizeof(mConstantBufferData));
   mConstantBuffers[m_currentFrame]->Unmap(0, nullptr);
}

void DirectXRenderer::UpdateBulletsBuffer() {
    const std::vector<BulletManager::BulletInstance>& instaces = mBulletMngr.getBulletsGPUData();
    if (!instaces.empty()) {
        void* data;
        mBulletsBuffer->Map(0, nullptr, &data);
        memcpy(data, instaces.data(), sizeof(BulletManager::BulletInstance) * instaces.size());
        mBulletsBuffer->Unmap(0, nullptr);
    }
}

void DirectXRenderer::Shutdown() {
    // graceful interruption of the thread
    mBulletsSpawnerThreadInterupter.clear(std::memory_order_relaxed);
    if (mBulletsSpawnerThread.joinable()) {
        mBulletsSpawnerThread.join();
    }
    // Drain the queue, wait for everything to finish
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        WaitForFence(mFrameFences[i].Get(), mFenceValues[i], mFrameFenceEvents[i]);
    }

    for (auto event : mFrameFenceEvents) {
        CloseHandle(event);
    }
}

// checking for G-SYNC or Free-Sync availability to avoid v-sync and posibble cpu blocking
bool DirectXRenderer::CheckTearingSupport() {
    BOOL allowTearing = FALSE;
#if defined(ALOW_TEARING)
    ComPtr<IDXGIFactory5> factory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        if (FAILED(factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing)))) {
            allowTearing = FALSE;
        }
    }
#endif
    return allowTearing;
}

void DirectXRenderer::CreateDeviceAndSwapChain() {
    // Enable the debug layers when in debug mode
    // you need get Graphics Tools installed to debug DirectX
#ifdef _DEBUG
    ComPtr<ID3D12Debug> debugController;
    D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
    debugController->EnableDebugLayer();
#endif

    DXGI_SWAP_CHAIN_DESC swapChainDesc;
    ::ZeroMemory(&swapChainDesc, sizeof(swapChainDesc));

    swapChainDesc.BufferCount = MAX_FRAMES_IN_FLIGHT;
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferDesc.Width = mWindow->width();
    swapChainDesc.BufferDesc.Height = mWindow->height();
    swapChainDesc.OutputWindow = mWindow->hwnd();
    swapChainDesc.SampleDesc.Count = 1;
    // DXGI_SWAP_CHAIN_DESC1 supports buffer\swapChain mismatch -> it will strech buffer to fit into swapChain
    // swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.Windowed = true;
    swapChainDesc.Flags = CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;  // set free-sync\g-sync instead v-sync

    // the driver may support directx 12 api but without hardware acceleration
    // D3D_FEATURE_LEVEL_11_0 hardware acceleration is present for sure
    auto renderEnv = CreateDeviceAndSwapChainHelper(D3D_FEATURE_LEVEL_11_0, &swapChainDesc);

    mDevice = renderEnv.device;
    mCommandQueue = renderEnv.queue;
    mSwapChain = renderEnv.swapChain;

    mRenderTargetViewDescriptorSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    SetupSwapChain();
}
