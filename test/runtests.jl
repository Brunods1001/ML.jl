using ML, Test


@testset "cost SSE" begin
    y = collect(1:100)
    ŷ = collect(1:100)
    cost = SSE(y, ŷ)
    @test cost == 0.0
end

