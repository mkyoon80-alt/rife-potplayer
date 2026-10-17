// Opt-in E04 layout experiment. All values remain FP32; this only changes the
// memory layout read by the existing four-candidate native-plane composition.
struct RifePackedMotion {
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11Buffer> flow,mask;
    ComPtr<ID3D11ShaderResourceView> flow_view,mask_view;
    ComPtr<ID3D11UnorderedAccessView> flow_out,mask_out;
    ComPtr<ID3D11ComputeShader> pack_shader;
    ComPtr<ID3D11PixelShader> y_shader,uv_shader;
    std::array<ID3D11ShaderResourceView*,PhaseCapacity+1> sources{};
    std::array<ComPtr<ID3D11ShaderResourceView>,PhaseCapacity+1> raw_sources;
    RifePackedMotion(ID3D11Device* device,ID3D11DeviceContext* ctx,ID3D11Buffer* motion,
                     ID3D11ShaderResourceView* first,const std::array<ComPtr<ID3D11ShaderResourceView>,PhaseCapacity>& batches):context(ctx){
        namespace packed=shaders::rife_packed;
        ck(device->CreateComputeShader(packed::packMotion,sizeof(packed::packMotion),nullptr,&pack_shader),"RIFE packed layout shader");
        ck(device->CreatePixelShader(packed::compositePackedY,sizeof(packed::compositePackedY),nullptr,&y_shader),"RIFE packed luma shader");
        ck(device->CreatePixelShader(packed::compositePackedUV,sizeof(packed::compositePackedUV),nullptr,&uv_shader),"RIFE packed chroma shader");
        auto buffer=[&](ComPtr<ID3D11Buffer>& b,unsigned bytes){D3D11_BUFFER_DESC d{};d.ByteWidth=bytes;d.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;ck(device->CreateBuffer(&d,nullptr,&b),"RIFE packed buffer");};
        buffer(flow,N*16);buffer(mask,N*4);
        D3D11_SHADER_RESOURCE_VIEW_DESC s{};s.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;s.Buffer.NumElements=N;
        s.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;ck(device->CreateShaderResourceView(flow.Get(),&s,&flow_view),"RIFE packed flow SRV");
        s.Format=DXGI_FORMAT_R32_FLOAT;ck(device->CreateShaderResourceView(mask.Get(),&s,&mask_view),"RIFE packed mask SRV");
        D3D11_UNORDERED_ACCESS_VIEW_DESC u{};u.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;u.Buffer.NumElements=N;
        u.Format=DXGI_FORMAT_R32G32B32A32_UINT;ck(device->CreateUnorderedAccessView(flow.Get(),&u,&flow_out),"RIFE packed flow UAV");
        u.Format=DXGI_FORMAT_R32_UINT;ck(device->CreateUnorderedAccessView(mask.Get(),&u,&mask_out),"RIFE packed mask UAV");
        sources[0]=first;for(unsigned i=0;i<PhaseCapacity;++i)sources[i+1]=batches[i].Get();
        for(unsigned i=0;i<sources.size();++i)if(sources[i]){sources[i]->GetDesc(&s);s.Format=DXGI_FORMAT_R32_UINT;ck(device->CreateShaderResourceView(motion,&s,&raw_sources[i]),"RIFE raw motion SRV");}
    }
    void pack(ID3D11ShaderResourceView* source,ID3D11Buffer* parameters){
        auto found=std::find(sources.begin(),sources.end(),source);
        if(found==sources.end())throw std::invalid_argument("Unknown packed motion source");
        auto raw=raw_sources[found-sources.begin()].Get();context->CSSetShaderResources(7,1,&raw);
        ID3D11UnorderedAccessView* u[]={flow_out.Get(),mask_out.Get()};context->CSSetUnorderedAccessViews(1,2,u,nullptr);
        context->CSSetConstantBuffers(0,1,&parameters);context->CSSetShader(pack_shader.Get(),nullptr,0);context->Dispatch((N+255)/256,1,1);
        raw=nullptr;context->CSSetShaderResources(7,1,&raw);u[0]=u[1]=nullptr;context->CSSetUnorderedAccessViews(1,2,u,nullptr);context->CSSetShader(nullptr,nullptr,0);
        ID3D11ShaderResourceView* views[]={flow_view.Get(),mask_view.Get()};context->PSSetShaderResources(5,2,views);
    }
    void unbind(){ID3D11ShaderResourceView* empty[2]{};context->PSSetShaderResources(5,2,empty);}
};
